import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { JobResponse } from "../types";

const POLL_INTERVAL_MS = 1000;

export function useJobPolling(jobId: string | null) {
  const [job, setJob] = useState<JobResponse | null>(null);
  const timeoutRef = useRef<number | null>(null);

  useEffect(() => {
    setJob(null);
    if (!jobId) return;

    let cancelled = false;

    async function poll() {
      try {
        const result = await api.getJob(jobId!);
        if (cancelled) return;
        setJob(result);
        if (result.status === "queued" || result.status === "running") {
          timeoutRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch {
        if (!cancelled) timeoutRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
      }
    }

    poll();

    return () => {
      cancelled = true;
      if (timeoutRef.current !== null) window.clearTimeout(timeoutRef.current);
    };
  }, [jobId]);

  return job;
}
