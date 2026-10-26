import type { Answer, Scheme } from "./types";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function errorMessage(res: Response): Promise<string> {
  if (res.status === 429) {
    return "The service has hit its usage quota. Please wait a minute and try again.";
  }
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg;
  } catch {
    // body was not JSON
  }
  return `Request failed (${res.status}). Please try again.`;
}

export async function fetchSchemes(): Promise<Scheme[]> {
  const res = await fetch("/api/schemes");
  if (!res.ok) throw new ApiError(await errorMessage(res), res.status);
  return res.json();
}

export async function askQuestion(question: string, scheme: string): Promise<Answer> {
  let res: Response;
  try {
    res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, scheme }),
    });
  } catch {
    throw new ApiError("Could not reach the server. Check your connection and try again.", 0);
  }
  if (!res.ok) throw new ApiError(await errorMessage(res), res.status);
  return res.json();
}
