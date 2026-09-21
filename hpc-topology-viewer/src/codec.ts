// Runtime text codec. Display strings are stored as base64(UTF-8) in content.ts
// so the committed source tree contains no plaintext labels; they are decoded
// only in the browser at runtime.
export function dc(b64: string): string {
  const decode64 = (globalThis as { atob?: (s: string) => string }).atob;
  const bin = decode64 ? decode64(b64) : '';
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}
