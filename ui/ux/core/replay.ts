// The hermetic replay on a compressed clock. `modelspec observe <recording> --serve 127.0.0.1:0
// --ui <dist>` serves the app and the recorded run; it sends the whole run at once. A pacing
// proxy in front of it passes every request through except `/events`, which it holds open and
// writes one recorded frame at a time when the harness says so, as a live `train --serve` stream
// would, so the app draws every state of the run and the gates read each; or, once started at a
// cadence, one frame every so many milliseconds whatever the page or the harness does.

import { type ChildProcess, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { type IncomingMessage, type Server, type ServerResponse, createServer } from 'node:http';
import type { AddressInfo } from 'node:net';

/** One SSE frame of the recorded stream: its `event:` name and its `data:` JSON. */
export interface Frame {
  text: string;
  event: string;
  data: unknown;
}

/** The frames of an SSE body, each with its terminating blank line. */
export function frames(body: string): Frame[] {
  return body
    .split(/\n\n/)
    .filter((chunk) => chunk.trim() !== '')
    .map((chunk) => {
      const event = /^event: (.*)$/m.exec(chunk)?.[1] ?? 'message';
      const data = /^data: (.*)$/m.exec(chunk)?.[1] ?? 'null';
      return { text: `${chunk}\n\n`, event, data: JSON.parse(data) as unknown };
    });
}

/** The frame of `data` under `event` with `id`, as the server writes it. */
export function frame(event: string, id: number, data: unknown): Frame {
  return { text: `event: ${event}\nid: ${id}\ndata: ${JSON.stringify(data)}\n\n`, event, data };
}

/** A running `modelspec observe`. */
export interface Origin {
  address: string;
  stop(): Promise<void>;
}

/** Starts `binary observe <recording> --serve 127.0.0.1:0 --ui <dist>` in `root`. */
export async function observe(binary: string, root: string, recording: string, dist: string): Promise<Origin> {
  if (!existsSync(binary)) {
    throw new Error(`${binary} does not exist: build it with \`cargo build -p modelspec-cli\``);
  }
  const child: ChildProcess = spawn(binary, ['observe', recording, '--serve', '127.0.0.1:0', '--ui', dist], {
    cwd: root,
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  const address = await new Promise<string>((resolve, reject) => {
    let seen = '';
    const timer = setTimeout(() => reject(new Error(`modelspec observe ${recording}: no serving line: ${seen}`)), 30_000);
    child.stderr?.setEncoding('utf8');
    child.stderr?.on('data', (chunk: string) => {
      seen += chunk;
      const found = /serving http:\/\/(\S+)\/events/.exec(seen);
      if (found?.[1] !== undefined) {
        clearTimeout(timer);
        resolve(found[1]);
      }
    });
    child.on('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`modelspec observe ${recording} exited ${code}: ${seen}`));
    });
  });
  return {
    address,
    stop: () =>
      new Promise<void>((resolve) => {
        if (child.exitCode !== null) {
          resolve();
          return;
        }
        child.once('exit', () => resolve());
        child.kill('SIGTERM');
      }),
  };
}

/** The recorded frames the origin serves at `/events`. */
export async function recorded(origin: Origin): Promise<Frame[]> {
  const response = await fetch(`http://${origin.address}/events`);
  return frames(await response.text());
}

/** The pacing proxy of one page. */
export interface Pacer {
  url: string;
  /** Resolves once the page has opened `/events`. */
  connected: Promise<void>;
  /** Writes the next `count` frames; ends the stream after the last. Returns how many remain. */
  push(count?: number): number;
  remaining(): number;
  /**
   * Writes one frame every `cadenceMs` from now until the last, on a timer of its own that waits
   * for nothing; resolves once the last frame is written.
   */
  start(cadenceMs: number): Promise<void>;
  /** How many frames had been written at `time` (`performance.now()` of this process). */
  sentAt(time: number): number;
  stop(): Promise<void>;
}

/** A proxy at a free loopback port in front of `origin` that paces `script` over `/events`. */
export async function pace(origin: Origin, script: readonly Frame[]): Promise<Pacer> {
  let next = 0;
  let timer: ReturnType<typeof setInterval> | null = null;
  const written: number[] = [];
  let stream: ServerResponse | null = null;
  let onConnect: () => void = () => undefined;
  const connected = new Promise<void>((resolve) => {
    onConnect = resolve;
  });
  const server: Server = createServer((request: IncomingMessage, response: ServerResponse) => {
    const path = request.url ?? '/';
    if (path.split('?')[0] === '/events') {
      if (stream !== null) {
        // The app reconnects only after an error; a second connection is a defect of the replay.
        response.writeHead(409).end();
        return;
      }
      response.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
      response.flushHeaders();
      stream = response;
      onConnect();
      return;
    }
    fetch(`http://${origin.address}${path}`, { method: request.method })
      .then(async (upstream) => {
        const body = Buffer.from(await upstream.arrayBuffer());
        response.writeHead(upstream.status, {
          'content-type': upstream.headers.get('content-type') ?? 'application/octet-stream',
          'content-length': body.length,
        });
        response.end(body);
      })
      .catch(() => response.writeHead(502).end());
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address() as AddressInfo;
  const push = (count = 1) => {
    if (stream === null) {
      throw new Error('push before the page opened /events');
    }
    const end = Math.min(script.length, next + count);
    stream.write(script.slice(next, end).map((f) => f.text).join(''));
    const now = performance.now();
    for (let i = next; i < end; i++) {
      written.push(now);
    }
    next = end;
    if (next === script.length) {
      stream.end();
    }
    return script.length - next;
  };
  return {
    url: `http://127.0.0.1:${port}/`,
    connected,
    push,
    remaining: () => script.length - next,
    start: (cadenceMs: number) =>
      new Promise<void>((resolve) => {
        if (next === script.length) {
          resolve();
          return;
        }
        timer = setInterval(() => {
          if (push(1) === 0 && timer !== null) {
            clearInterval(timer);
            timer = null;
            resolve();
          }
        }, cadenceMs);
      }),
    sentAt: (time: number) => written.filter((t) => t <= time).length,
    stop: () =>
      new Promise<void>((resolve) => {
        if (timer !== null) {
          clearInterval(timer);
          timer = null;
        }
        stream?.destroy();
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}
