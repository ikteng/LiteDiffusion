import type { HistoryItem } from "../types";

const DB_NAME = "minimax-h3-studio";
const DB_VERSION = 1;
const STORE = "clips";
const MAX_CLIPS = 12;
const MAX_BYTES = 1024 ** 3;

type StoredClip = Omit<HistoryItem, "url"> & { video: Blob };

function request<T>(value: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    value.onsuccess = () => resolve(value.result);
    value.onerror = () => reject(value.error ?? new Error("Browser history storage failed."));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("Browser history transaction failed."));
    transaction.onabort = () => reject(transaction.error ?? new Error("Browser history transaction was aborted."));
  });
}

function openHistory(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in window)) {
      reject(new Error("IndexedDB is unavailable."));
      return;
    }
    const opening = window.indexedDB.open(DB_NAME, DB_VERSION);
    opening.onupgradeneeded = () => {
      if (!opening.result.objectStoreNames.contains(STORE)) {
        opening.result.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    opening.onsuccess = () => resolve(opening.result);
    opening.onerror = () => reject(opening.error ?? new Error("Could not open browser history."));
  });
}

async function storedClips(database: IDBDatabase): Promise<StoredClip[]> {
  const transaction = database.transaction(STORE, "readonly");
  const clips = await request(transaction.objectStore(STORE).getAll() as IDBRequest<StoredClip[]>);
  await transactionDone(transaction);
  return clips.sort((a, b) => b.createdAt - a.createdAt);
}

export async function restoreHistory(): Promise<HistoryItem[]> {
  let database: IDBDatabase | null = null;
  try {
    database = await openHistory();
    return (await storedClips(database)).map(({ video, ...item }) => ({
      ...item,
      url: URL.createObjectURL(video),
    }));
  } catch {
    return [];
  } finally {
    database?.close();
  }
}

async function deleteIds(database: IDBDatabase, ids: string[]): Promise<void> {
  if (ids.length === 0) return;
  const transaction = database.transaction(STORE, "readwrite");
  const store = transaction.objectStore(STORE);
  for (const id of ids) store.delete(id);
  await transactionDone(transaction);
}

function clipBytes(clip: Pick<StoredClip, "video" | "sourceImage" | "sourceLastImage">): number {
  return clip.video.size + (clip.sourceImage?.size ?? 0) + (clip.sourceLastImage?.size ?? 0);
}

/** Save the actual MP4, pruning before the write so one large clip cannot grow storage without bound. */
export async function saveHistoryItem(item: HistoryItem): Promise<void> {
  const response = await fetch(item.url);
  if (!response.ok) throw new Error(`Could not save generated video (${response.status}).`);
  const video = await response.blob();
  if (video.size === 0 || video.size > MAX_BYTES) throw new Error("Generated video is too large for browser history.");

  const database = await openHistory();
  try {
    const existing = (await storedClips(database)).filter((clip) => clip.id !== item.id);
    let bytes = video.size + (item.sourceImage?.size ?? 0) + (item.sourceLastImage?.size ?? 0);
    const keep: StoredClip[] = [];
    for (const clip of existing) {
      if (keep.length + 1 >= MAX_CLIPS || bytes + clipBytes(clip) > MAX_BYTES) continue;
      keep.push(clip);
      bytes += clipBytes(clip);
    }
    await deleteIds(database, existing.filter((clip) => !keep.includes(clip)).map((clip) => clip.id));

    const { url: _url, ...metadata } = item;
    const transaction = database.transaction(STORE, "readwrite");
    transaction.objectStore(STORE).put({ ...metadata, video } satisfies StoredClip);
    await transactionDone(transaction);
  } finally {
    database.close();
  }
}

export async function deleteHistoryItem(id: string): Promise<void> {
  let database: IDBDatabase | null = null;
  try {
    database = await openHistory();
    await deleteIds(database, [id]);
  } catch {
    // A failed durable delete must not prevent the in-memory history control from working.
  } finally {
    database?.close();
  }
}
