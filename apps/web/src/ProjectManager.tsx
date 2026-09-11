import { useState } from "react";
import type { Project } from "./types";
export function ProjectManager({
  projects,
  busy,
  mutate,
}: {
  projects: Project[];
  busy: boolean;
  mutate: (tool: string, args: unknown, message: string) => Promise<unknown>;
}) {
  const [name, setName] = useState("");
  return (
    <details className="project-manager">
      <summary>
        Manage projects{" "}
        <span>{projects.filter((p) => !p.archived).length}</span>
      </summary>
      <form
        className="project-create"
        onSubmit={async (e) => {
          e.preventDefault();
          if (await mutate("project.create", { name }, "Project created"))
            setName("");
        }}
      >
        <input
          aria-label="New project name"
          required
          maxLength={200}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="New project name"
        />
        <button className="secondary" disabled={busy || !name.trim()}>
          Create project
        </button>
      </form>
      {projects.map((p) => (
        <form
          key={p.id + ":" + p.revision}
          className="project-editor"
          onSubmit={async (e) => {
            e.preventDefault();
            const data = new FormData(e.currentTarget);
            await mutate(
              "project.update",
              {
                project_id: p.id,
                expected_revision: p.revision,
                name: String(data.get("name")),
                description: String(data.get("description")),
              },
              "Project saved",
            );
          }}
        >
          <input
            name="name"
            aria-label={"Project name: " + p.name}
            defaultValue={p.name}
            required
            maxLength={200}
          />
          <input
            name="description"
            aria-label={"Project description: " + p.name}
            defaultValue={p.description}
            placeholder="Description"
            maxLength={10000}
          />
          <button className="secondary compact" disabled={busy}>
            Save
          </button>
          <button
            type="button"
            className="text-button"
            disabled={busy}
            onClick={() =>
              void mutate(
                "project.update",
                {
                  project_id: p.id,
                  expected_revision: p.revision,
                  archived: !p.archived,
                },
                p.archived ? "Project restored" : "Project archived",
              )
            }
          >
            {p.archived ? "Restore" : "Archive"}
          </button>
        </form>
      ))}
      <p className="footnote">
        Archiving a project preserves its tasks and reminders.
      </p>
    </details>
  );
}
