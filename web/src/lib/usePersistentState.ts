import { useEffect, useState } from "react";

/** useState that survives remounts and reloads (this browser only; best effort). */
export function usePersistentState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = window.localStorage.getItem(key);
      return stored === null ? initial : (JSON.parse(stored) as T);
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Storage may be unavailable (private mode); the value still holds for this mount.
    }
  }, [key, value]);
  return [value, setValue] as const;
}
