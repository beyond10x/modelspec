// What a round reads from git and from the served files: the main branch's baseline, the commit
// a round measured and whether its tree was clean, and the hash of the served app. Free of any
// project code, like the rest of `core/`.

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';

function git(root: string, args: string[]): string {
  return execFileSync('git', args, {
    cwd: root,
    env: { ...process.env, GIT_OPTIONAL_LOCKS: '0' },
    stdio: ['ignore', 'pipe', 'pipe'],
  }).toString();
}

/**
 * The JSON at `path` on `ref` (`origin/main`): `null` when `ref` exists and has no such file (the
 * first landing); an error naming `ref` when the ref itself is missing, as in a depth-1 checkout
 * that never fetched it, because the rule that a baseline is never made worse than main's cannot
 * run without it.
 */
export function mainBaseline(root: string, path: string, ref = 'origin/main'): unknown {
  try {
    git(root, ['rev-parse', '--verify', '--quiet', `${ref}^{commit}`]);
  } catch {
    throw new Error(
      `the ratchet needs ${ref} to compare baseline.json with main's, and ${ref} is missing: git fetch --no-tags origin main:refs/remotes/${ref}`,
    );
  }
  try {
    git(root, ['cat-file', '-e', `${ref}:${path}`]);
  } catch {
    return null;
  }
  return JSON.parse(git(root, ['show', `${ref}:${path}`])) as unknown;
}

/** The commit a round measured: HEAD, and whether the tree differs from it outside `exclude`. */
export function commitInfo(root: string, exclude: readonly string[]): { head: string; dirty: boolean } {
  try {
    const head = git(root, ['rev-parse', 'HEAD']).trim();
    const status = git(root, ['status', '--porcelain', '--', '.', ...exclude.map((p) => `:(exclude)${p}`)]);
    return { head, dirty: status.trim() !== '' };
  } catch {
    return { head: 'unknown', dirty: true };
  }
}

/** Every file under `dir`, relative and sorted. */
function files(dir: string, prefix = ''): string[] {
  return readdirSync(`${dir}/${prefix}`, { withFileTypes: true })
    .flatMap((e) => (e.isDirectory() ? files(dir, `${prefix}${e.name}/`) : [`${prefix}${e.name}`]))
    .sort();
}

/**
 * The SHA-256 of `dir`: over its sorted relative file list, each path followed by the SHA-256 of
 * its contents, one line each; `missing` when there is no such directory.
 */
export function treeSha256(dir: string): string {
  let list: string[];
  try {
    list = files(dir);
  } catch {
    return 'missing';
  }
  const hash = createHash('sha256');
  for (const name of list) {
    hash.update(`${name}\0${createHash('sha256').update(readFileSync(`${dir}/${name}`)).digest('hex')}\n`);
  }
  return hash.digest('hex');
}
