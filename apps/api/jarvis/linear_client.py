"""Linear's public GraphQL API. No Slack/agent relay or browser credentials."""

import httpx


class LinearFailure(Exception):
    def __init__(self, code="unavailable", retry=False, unknown=False):
        self.code, self.retry, self.unknown = code, retry, unknown
        super().__init__(code)


ISSUE_FIELDS = """id identifier url title description dueDate priority updatedAt archivedAt
state { id name type } team { id name key } assignee { id name }
project { id name description } parent { id } labels(first:100) { nodes { id name } pageInfo { hasNextPage } }"""
ISSUE_QUERY = "query EriIssue($id:String!){issue(id:$id){" + ISSUE_FIELDS + "}}"
ISSUES_QUERY = (
    "query EriIssues($filter:IssueFilter,$after:String){issues(first:100,after:$after,filter:$filter,includeArchived:true,orderBy:updatedAt){nodes{"
    + ISSUE_FIELDS
    + "} pageInfo{hasNextPage endCursor}}}"
)
CREATE_QUERY = (
    "mutation EriCreate($input:IssueCreateInput!){issueCreate(input:$input){success issue{"
    + ISSUE_FIELDS
    + "}}}"
)
UPDATE_QUERY = (
    "mutation EriUpdate($id:String!,$input:IssueUpdateInput!){issueUpdate(id:$id,input:$input){success issue{"
    + ISSUE_FIELDS
    + "}}}"
)
# The completion comment of an agent run (agent_sync.py). Linear has no idempotency key for commentCreate,
# so a write whose response was lost is reconciled by finding the run marker in the issue's comments.
COMMENT_QUERY = "mutation EriComment($input:CommentCreateInput!){commentCreate(input:$input){success comment{id url body}}}"
COMMENTS_QUERY = (
    "query EriComments($id:String!,$after:String){issue(id:$id){comments(first:100,after:$after){nodes{id url body}"
    " pageInfo{hasNextPage endCursor}}}}"
)


class LinearClient:
    def __init__(self, key):
        from .config import require_external_services
        require_external_services()
        self.http = httpx.Client(
            base_url="https://api.linear.app", headers={"Authorization": key}, timeout=25
        )

    def close(self):
        self.http.close()

    def query(self, query, variables=None):
        writing = query.lstrip().startswith("mutation")
        try:
            response = self.http.post("/graphql", json={"query": query, "variables": variables or {}})
            body = response.json()
        except (httpx.HTTPError, ValueError):
            raise LinearFailure("unavailable", retry=True, unknown=writing) from None
        if not isinstance(body, dict):
            raise LinearFailure("invalid_response", retry=True, unknown=writing)
        errors = body.get("errors", [])
        if not isinstance(errors, list) or any(not isinstance(e, dict) for e in errors):
            raise LinearFailure("invalid_response", retry=True, unknown=writing)
        if errors:
            codes = {str(e.get("extensions", {}).get("code", "")).upper() for e in errors}
            if codes & {"ENTITY_NOT_FOUND", "NOT_FOUND"}:
                raise LinearFailure("not_found")
            if "RATELIMITED" in codes:
                raise LinearFailure("rate_limited", retry=True)
            if codes & {"AUTHENTICATION_ERROR", "AUTHENTICATION_REQUIRED", "FORBIDDEN"}:
                raise LinearFailure("permission")
            raise LinearFailure("rejected", unknown=writing)
        if response.status_code >= 400:
            raise LinearFailure(
                "permission" if response.status_code in {401, 403} else "unavailable",
                retry=response.status_code == 429 or response.status_code >= 500,
                unknown=writing and response.status_code >= 500,
            )
        if not isinstance(body.get("data"), dict):
            raise LinearFailure("invalid_response", retry=True, unknown=writing)
        return body["data"]

    def pages(self, query, field, variables=None, max_pages=100):
        out = []
        cursor = None
        seen = set()
        for _ in range(max_pages):
            data = self.query(query, {**(variables or {}), "after": cursor})[field]
            out.extend(data["nodes"])
            if not data["pageInfo"]["hasNextPage"]:
                return out
            cursor = data["pageInfo"].get("endCursor")
            if not cursor or cursor in seen:
                break
            seen.add(cursor)
        raise LinearFailure("range_too_large")

    def issue(self, remote_id):
        try:
            return self.query(ISSUE_QUERY, {"id": remote_id}).get("issue")
        except LinearFailure as error:
            if error.code == "not_found":
                return None
            raise

    def comment(self, issue_id, body):
        saved = self.query(COMMENT_QUERY, {"input": {"issueId": issue_id, "body": body}})["commentCreate"]
        if not saved.get("success") or not (saved.get("comment") or {}).get("id"):
            raise LinearFailure("invalid_response", retry=True, unknown=True)
        return saved["comment"]

    def comments(self, issue_id, max_pages=20):
        out, cursor = [], None
        for _ in range(max_pages):
            issue = self.query(COMMENTS_QUERY, {"id": issue_id, "after": cursor}).get("issue")
            if issue is None:
                raise LinearFailure("not_found")
            page = issue["comments"]
            out.extend(page["nodes"])
            if not page["pageInfo"]["hasNextPage"] or not page["pageInfo"].get("endCursor"):
                return out
            cursor = page["pageInfo"]["endCursor"]
        raise LinearFailure("range_too_large")

    def identity(self):
        result = self.query("query EriIdentity{viewer{id name} organization{id name}}")
        teams = self.pages(
            "query EriTeams($after:String){teams(first:100,after:$after){nodes{id name key} pageInfo{hasNextPage endCursor}}}",
            "teams",
            max_pages=10,
        )
        return {**result, "teams": teams}

    def directory(self, team_ids):
        states = self.pages(
            "query EriStates($after:String){workflowStates(first:100,after:$after){nodes{id name type team{id}} pageInfo{hasNextPage endCursor}}}",
            "workflowStates",
            max_pages=20,
        )
        users = self.pages(
            "query EriUsers($after:String){users(first:100,after:$after){nodes{id name active} pageInfo{hasNextPage endCursor}}}",
            "users",
            max_pages=20,
        )
        return {"states": [s for s in states if s["team"]["id"] in team_ids], "users": users}

    def issues(self, filter):
        return self.pages(ISSUES_QUERY, "issues", {"filter": filter})
