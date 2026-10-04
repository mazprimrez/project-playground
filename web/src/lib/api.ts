import { useEffect, useState } from "react";

/** The server's list of mounted projects (GET /api/). */
export type ProjectStatus = { name: string; title: string | null; status: "up" | "down"; docs: string };

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, body?.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

type State<T> = { data: T | null; error: string | null; loading: boolean };

/** Fetch JSON when `url` changes (null = don't fetch); the previous request is cancelled. */
export function useJson<T>(url: string | null): State<T> {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: url !== null });
  useEffect(() => {
    if (url === null) return;
    const ctrl = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));
    getJson<T>(url, ctrl.signal)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((e) => {
        if (!ctrl.signal.aborted) setState({ data: null, error: String(e.message ?? e), loading: false });
      });
    return () => ctrl.abort();
  }, [url]);
  return state;
}
