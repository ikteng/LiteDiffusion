export function formatModelSize(mb: number): string {
  if (!mb) return "";
  if (mb >= 1024) return `${(mb / 1024).toFixed(1).replace(/\.0$/, "")} GB`;
  return `${mb} MB`;
}

export function sortModelsBySize<T extends { approx_size_mb: number }>(
  models: T[],
): T[] {
  return [...models].sort((a, b) => a.approx_size_mb - b.approx_size_mb);
}
