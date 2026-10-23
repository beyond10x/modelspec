import type { RunEvent, RunEventBody } from '../generated/observatory.ts';
import { ends } from './series.ts';

/** Every `event:` name the stream sends; a kind added in Rust and missing here fails the type check. */
const KINDS: Record<RunEventBody['kind'], null> = {
  RunStarted: null,
  StepCompleted: null,
  RunFinished: null,
  RunFailed: null,
};

/** How long the page waits after a dropped connection before it connects again, in ms. */
export const RETRY_MS = 1000;

export interface StreamHandlers {
  event(event: RunEvent): void;
  /** The connection dropped before the run ended; the page connects again after `RETRY_MS`. */
  interrupted(): void;
}

/**
 * Opens the run's event stream at `url` (relative to the page, so `events` beside the served UI)
 * and hands each event to `handlers`. The source is closed on `RunFinished` or `RunFailed`: the
 * server ends the stream there and answers a reconnect with an empty response. Returns the close.
 *
 * A dropped connection is not left to the browser's own reconnect, which sends `Last-Event-ID`:
 * a server restarted at the same address serves another run, and from that id on it has at most
 * the tail of that run. The page opens a new source instead, which reads the stream from its
 * start, so it gets the new run's `RunStarted`; `apply` skips the events of the same run it has
 * already folded.
 */
export function subscribe(url: string, handlers: StreamHandlers): () => void {
  let source: EventSource | null = null;
  let retry: ReturnType<typeof setTimeout> | undefined;
  let done = false;
  const close = () => {
    done = true;
    clearTimeout(retry);
    source?.close();
  };
  const open = () => {
    const current = new EventSource(url);
    source = current;
    const listener = (message: MessageEvent<string>) => {
      const event = JSON.parse(message.data) as RunEvent;
      handlers.event(event);
      if (ends(event)) {
        close();
      }
    };
    for (const kind of Object.keys(KINDS)) {
      current.addEventListener(kind, listener);
    }
    current.onerror = () => {
      current.close();
      if (done) {
        return;
      }
      handlers.interrupted();
      retry = setTimeout(open, RETRY_MS);
    };
  };
  open();
  return close;
}
