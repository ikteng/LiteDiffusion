import { useEffect, useState } from "react";
import { api } from "../api";

export function useModelDownloads() {
  const [statuses, setStatuses] = useState<Record<string, string>>({});

  useEffect(() => {
    api.getModelStatus().then(setStatuses);
  }, []);

  const downloadModel = async (key: string) => {
    await api.downloadModel(key);
    setStatuses((prev) => ({ ...prev, [key]: "downloading" }));
  };

  const getStatus = (key: string) => {
    if (statuses[key]) return statuses[key];
    return "idle";
  };

  return { downloadModel, getStatus };
}
