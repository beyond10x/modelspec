import type { RunEvent, RunEventBody } from '../generated/observatory.ts';
import { ends } from './series.ts';

/** Every `event:` name the stream sends; a kind added in Rust and missing here fails the type check. */
const KINDS: Record<RunEventBody['kind'], null> = {
  RunStarted: null,
  StepCompleted: null,
  RunFinished: null,
  RunFailed: null,
};

export interface StreamHandlers {
  event(event: RunEvent): void;
  /** The connection dropped before the run ended; the browser reconnects with `Last-Event-ID`. */
  interrupted(): void;
}

/**
 * Opens the run's event stream at `url` (relative to the page, so `events` beside the served UI)
 * and hands each event to `handlers`. The source is closed on `RunFinished` or `RunFailed`: the
 * server ends the stream there and answers a reconnect with an empty response. Returns the close.
 */
export function subscribe(url: string, handlers: StreamHandlers): () => void {
  const source = new EventSource(url);
  const listener = (message: MessageEvent<string>) => {
    const event = JSON.parse(message.data) as RunEvent;
    handlers.event(event);
    if (ends(event)) {
      source.close();
    }
  };
  for (const kind of Object.keys(KINDS)) {
    source.addEventListener(kind, listener);
  }
  source.onerror = () => {
    if (source.readyState !== EventSource.CLOSED) {
      handlers.interrupted();
    }
  };
  return () => source.close();
}
