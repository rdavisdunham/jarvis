"""Shared lexical canonical form: spoken numbers, punctuation and Unicode variants compare equal.

canon("Test, test, one, two, three") == canon("test test 123") == "test test 123"
canon("Route sixty-six") == canon("route 66") == "route 66"
Use canon() on BOTH sides of any Python comparison, sql_filter() for legacy substring
searches over raw columns, and tsquery() for full-text search over raw content.
"""

import re
import unicodedata

ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
UNIT = {w: i for i, w in enumerate(ONES)}
TEN = {w: 20 + 10 * i for i, w in enumerate(TENS)}


def words(text):
    """NFKC + casefold word tokens; punctuation and underscores separate tokens."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)  # 1,000 -> 1000
    text = re.sub(r"(?<=\d)\.(?=\d)", " point ", text)  # 3.5 -> 3 point 5, never 35
    return re.findall(r"[^\W_]+", text)


def plain(text):
    """Punctuation/case-insensitive form that leaves number words untouched."""
    return " ".join(words(text))


def _small(tokens, i):
    if i < len(tokens) and tokens[i] in TEN:
        value, i = TEN[tokens[i]], i + 1
        if i < len(tokens) and 0 < UNIT.get(tokens[i], 0) < 10:
            value, i = value + UNIT[tokens[i]], i + 1
        return value, i
    if i < len(tokens) and tokens[i] in UNIT:
        return UNIT[tokens[i]], i + 1
    return None, i


def _group(tokens, i):
    value, j = _small(tokens, i)
    if value and value < 100 and j < len(tokens) and tokens[j] == "hundred":
        value, j = value * 100, j + 1
        k = j + 1 if j < len(tokens) and tokens[j] == "and" else j
        rest, k = _small(tokens, k)
        if rest is not None:
            value, j = value + rest, k
    return value, j


def _number(tokens, i):
    value, j = _group(tokens, i)
    if value and j < len(tokens) and tokens[j] == "thousand":
        value, j = value * 1000, j + 1
        k = j + 1 if j < len(tokens) and tokens[j] == "and" else j
        rest, k = _group(tokens, k)
        if rest is not None and rest < 1000:
            value, j = value + rest, k
    return value, j


def canon_tokens(text):
    tokens, out, i = words(text), [], 0
    while i < len(tokens):
        value, j = _number(tokens, i)
        if value is None:
            out.append(tokens[i])
            i += 1
        else:
            out.append(str(value))
            i = j
    return _join_digits(out)


def _join_digits(tokens):
    result, run = [], False
    for token in tokens:
        single = len(token) == 1 and token.isdigit()
        if single and run:
            result[-1] += token
        else:
            result.append(token)
        run = single
    return result


def canon(text):
    return " ".join(canon_tokens(text))


def spoken(value):
    if value < 20:
        return ONES[value]
    if value < 100:
        return TENS[value // 10 - 2] + ("" if value % 10 == 0 else " " + ONES[value % 10])
    if value < 1000:
        return ONES[value // 100] + " hundred" + ("" if value % 100 == 0 else " " + spoken(value % 100))
    return spoken(value // 1000) + " thousand" + ("" if value % 1000 == 0 else " " + spoken(value % 1000))


def variants(token):
    """Raw-text spellings of one canonical token: 66 -> 66, 6 6, six six, sixty six."""
    if not token.isdigit():
        return [token]
    out = [token, " ".join(ONES[int(d)] for d in token)]
    if len(token) > 1:
        out.append(" ".join(token))
    if not (len(token) > 1 and token[0] == "0") and len(token) <= 6:
        out.append(spoken(int(token)))
    return list(dict.fromkeys(out))


def pattern(token):
    """PostgreSQL ARE (~*) matching any raw spelling; the raw token keeps substring semantics."""
    alternatives = [token] + [r"\m" + r"\W+".join(v.split()) + r"\M" for v in variants(token)[1:]]
    return "(" + "|".join(alternatives) + ")"


def sql_filter(query, *columns, limit=12):
    """Legacy literal substring match OR every canonical token matched in any spelling."""
    from sqlalchemy import Text, and_, cast, func, or_

    term = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    raw = [cast(c, Text).ilike("%" + term + "%", escape="\\") for c in columns]
    tokens = canon_tokens(query)[:limit]
    if not tokens:
        return or_(*raw)
    haystack = func.concat_ws(" ", *[cast(c, Text) for c in columns])
    return or_(*raw, and_(*[haystack.op("~*")(pattern(t)) for t in tokens]))


def tsquery(query, config="english", limit=12):
    """AND across canonical tokens, OR across raw spellings of each token. None when empty."""
    from sqlalchemy import func

    result = None
    for token in canon_tokens(query)[:limit]:
        alternatives = None
        for spelling in variants(token):
            q = func.plainto_tsquery(config, spelling)
            alternatives = q if alternatives is None else alternatives.op("||")(q)
        result = alternatives if result is None else result.op("&&")(alternatives)
    return result
