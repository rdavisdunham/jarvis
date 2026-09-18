// Whole-project definition: review all deletions; keep secrets in Railway.
// Version pins are intentional. Do not use the postgres() default on an existing volume.
import { bucket, defineRailway, github, postgres, preserve, project, service, volume } from "railway/iac";

export default defineRailway(() => {
  const Postgres16 = postgres("Postgres16", { image: "ghcr.io/railwayapp-templates/postgres-ssl:16", region: "us-west2" });
  Postgres16.deploy = { ...Postgres16.deploy, overlapSeconds: 0, restartPolicyType: "ALWAYS" };
  Postgres16.networking = { privateNetworkEndpoint: "postgres16" };
  const Postgres = postgres("Postgres", { image: "ghcr.io/railwayapp-templates/postgres-ssl:18", region: "us-west2" });
  Postgres.networking = { privateNetworkEndpoint: "postgres" };
  const postgres16Volume = volume("postgres16-volume", { alerts: { usage: { "100": {}, "80": {}, "95": {} } }, allowOnlineResize: true, region: "us-west2", sizeMB: 5000 });
  const postgresVolume = volume("postgres-volume", { alerts: { usage: { "100": {}, "80": {}, "95": {} } }, allowOnlineResize: true, region: "us-west2", sizeMB: 5000 });
  const PostgresPITR = bucket("Postgres-PITR", { region: "sjc" });
  const Eridani_Web = service("Eridani_Web", {
    source: github("rdavisdunham/jarvis", { branch: "main" }),
    build: { buildEnvironment: "V3", builder: "DOCKERFILE", dockerfilePath: "Dockerfile.upgrade" },
    start: "/app/.venv/bin/python -m jarvis.deploy api",
    healthcheck: "/health/ready",
    healthcheckTimeout: 180,
    preDeploy: "/app/.venv/bin/python -m jarvis.deploy migrate",
    replicas: { "us-west2": 1 },
    deploy: { drainingSeconds: 30, overlapSeconds: 0, restartPolicyType: "ALWAYS" },
    domains: [{ domain: "app.eridani.app", port: 8765 }],
    networking: { privateNetworkEndpoint: "jarvis" },
    env: { FORWARDED_ALLOW_IPS: preserve(), JARVIS_COST_TRACKING_ENABLED: preserve(), JARVIS_DATABASE_MAX_OVERFLOW: preserve(), JARVIS_DATABASE_POOL_SIZE: preserve(), JARVIS_DATABASE_URL: preserve(), JARVIS_DBOS_CLIENT_POOL_SIZE: preserve(), JARVIS_DBOS_POOL_SIZE: preserve(), JARVIS_DEPLOYMENT_ENVIRONMENT: preserve(), JARVIS_ENV_FILE: preserve(), JARVIS_EXTERNAL_SERVICES_ENABLED: preserve(), JARVIS_GEMINI_API_KEY: preserve(), JARVIS_GOOGLE_CLIENT_ID: preserve(), JARVIS_GOOGLE_CLIENT_SECRET: preserve(), JARVIS_INTEGRATION_ENCRYPTION_KEY: preserve(), JARVIS_MAINTENANCE_MODE: preserve(), JARVIS_OPENAI_API_KEY: preserve(), JARVIS_ORIGIN: preserve(), JARVIS_OWNER_ID: preserve(), JARVIS_OWNER_NAME: preserve(), JARVIS_PAIRING_ENABLED: preserve(), JARVIS_SEMANTIC_SEARCH_ENABLED: preserve(), JARVIS_TIMEZONE: preserve(), JARVIS_VAPID_PRIVATE_KEY: preserve(), JARVIS_VAPID_PUBLIC_KEY: preserve(), JARVIS_VAPID_SUBJECT: preserve(), JARVIS_WORKER_ENABLED: preserve(), PORT: preserve(), WEB_CONCURRENCY: preserve() },
  });
  const Eridani_Worker = service("Eridani_Worker", {
    source: github("rdavisdunham/jarvis", { branch: "main" }),
    build: { buildEnvironment: "V3", builder: "DOCKERFILE", dockerfilePath: "Dockerfile.upgrade" },
    start: "/app/.venv/bin/python -m jarvis.deploy worker",
    preDeploy: "/app/.venv/bin/python -m jarvis.deploy migrate",
    replicas: { "us-west2": 1 },
    deploy: { drainingSeconds: 30, overlapSeconds: 0, restartPolicyType: "ALWAYS" },
    networking: { privateNetworkEndpoint: "eridaniworker" },
    env: { FORWARDED_ALLOW_IPS: preserve(), JARVIS_COST_TRACKING_ENABLED: preserve(), JARVIS_DATABASE_MAX_OVERFLOW: preserve(), JARVIS_DATABASE_POOL_SIZE: preserve(), JARVIS_DATABASE_URL: preserve(), JARVIS_DBOS_CLIENT_POOL_SIZE: preserve(), JARVIS_DBOS_POOL_SIZE: preserve(), JARVIS_DEPLOYMENT_ENVIRONMENT: preserve(), JARVIS_ENV_FILE: preserve(), JARVIS_EXTERNAL_SERVICES_ENABLED: preserve(), JARVIS_GEMINI_API_KEY: preserve(), JARVIS_GOOGLE_CLIENT_ID: preserve(), JARVIS_GOOGLE_CLIENT_SECRET: preserve(), JARVIS_INTEGRATION_ENCRYPTION_KEY: preserve(), JARVIS_MAINTENANCE_MODE: preserve(), JARVIS_OPENAI_API_KEY: preserve(), JARVIS_ORIGIN: preserve(), JARVIS_OWNER_ID: preserve(), JARVIS_OWNER_NAME: preserve(), JARVIS_PAIRING_ENABLED: preserve(), JARVIS_SEMANTIC_SEARCH_ENABLED: preserve(), JARVIS_TIMEZONE: preserve(), JARVIS_VAPID_PRIVATE_KEY: preserve(), JARVIS_VAPID_PUBLIC_KEY: preserve(), JARVIS_VAPID_SUBJECT: preserve(), JARVIS_WORKER_ENABLED: preserve(), PORT: preserve(), WEB_CONCURRENCY: preserve() },
  });

  return project("Eridani", {
    resources: [Eridani_Web, Eridani_Worker, Postgres16, Postgres, postgres16Volume, postgresVolume, PostgresPITR],
  });
});
