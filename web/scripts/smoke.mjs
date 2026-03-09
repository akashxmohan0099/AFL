import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";

const port = process.env.SMOKE_PORT || "3105";
const host = "127.0.0.1";
const baseUrl = `http://${host}:${port}`;
const routes = ["/", "/schedule", "/matches", "/players", "/odds", "/predictions"];

function isReady(output) {
  return output.includes("Ready") || output.includes(`http://${host}:${port}`);
}

async function waitForServer() {
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${baseUrl}/`);
      if (res.ok) {
        return;
      }
    } catch {
      // Server not ready yet.
    }
    await delay(250);
  }
  throw new Error("Timed out waiting for Next.js server");
}

async function main() {
  const child = spawn(
    "npm",
    ["run", "start", "--", "--hostname", host, "--port", port],
    {
      env: { ...process.env, PORT: port },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  let output = "";
  child.stdout.on("data", (chunk) => {
    output += chunk.toString();
  });
  child.stderr.on("data", (chunk) => {
    output += chunk.toString();
  });

  const childExit = new Promise((_, reject) => {
    child.once("exit", (code, signal) => {
      reject(
        new Error(
          `Next.js server exited before smoke completed (code=${code ?? "null"}, signal=${signal ?? "null"})\n${output}`,
        ),
      );
    });
  });

  try {
    await Promise.race([waitForServer(), childExit]);

    for (const route of routes) {
      const res = await fetch(`${baseUrl}${route}`);
      if (!res.ok) {
        throw new Error(`Route ${route} returned ${res.status}`);
      }
      const contentType = res.headers.get("content-type") || "";
      if (!contentType.includes("text/html")) {
        throw new Error(`Route ${route} returned unexpected content-type: ${contentType}`);
      }
    }

    console.log(`Smoke passed for ${routes.length} routes on ${baseUrl}`);
  } finally {
    child.kill("SIGTERM");
    await delay(250);
    if (!child.killed) {
      child.kill("SIGKILL");
    }
  }

  if (!isReady(output)) {
    throw new Error(`Next.js startup output was unexpected:\n${output}`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
