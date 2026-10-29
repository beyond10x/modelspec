import {useEffect, useRef, type CSSProperties, type ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';

type DiagramView = {
  root: string;
  levels: Record<string, string>;
};

export type DiagramModule = {
  program: string;
  source: string;
  views: Record<string, DiagramView>;
};

type ModelDiagramProps = {
  data: DiagramModule;
  view: string;
  scope?: string;
  caption?: ReactNode;
  pageLink?: boolean;
};

export default function ModelDiagram({data, view, scope, caption, pageLink = true}: ModelDiagramProps): ReactNode {
  const canvas = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = canvas.current;
    if (element !== null && element.scrollWidth > element.clientWidth) {
      element.scrollLeft = (element.scrollWidth - element.clientWidth) / 2;
    }
  }, []);
  const drawn = data.views[view];
  const key = scope ?? drawn?.root;
  const standalone = useBaseUrl(`/img/diagrams/${data.program}/${view}/${key}.svg`);
  const svg = key === undefined ? undefined : drawn?.levels[key];
  if (svg === undefined) {
    throw new Error(`${data.program} has no level ${key ?? '(root)'} in its ${view} view`);
  }
  const width = /^<svg[^>]* width="(\d+)"/.exec(svg)?.[1];
  const style = width === undefined ? undefined : ({'--ms-diagram-width': width} as CSSProperties);
  return (
    <figure className="ms-figure">
      <div ref={canvas} className="ms-figure__canvas" style={style} tabIndex={0} dangerouslySetInnerHTML={{__html: svg}} />
      <figcaption>
        {caption ?? (
          <>
            The {view} view of <code>{key}</code>, drawn by <code>modelspec-diagram</code> from its
            verified Tensor IR.
          </>
        )}{' '}
        <a href={standalone}>Open the SVG</a>
        {pageLink ? (
          <>
            {' or '}
            <Link to={`/docs/diagrams/${data.program}`}>every level of {data.program}</Link>
          </>
        ) : null}
        .
      </figcaption>
    </figure>
  );
}

type GlyphProps = {
  id: string;
  label: string;
  small?: boolean;
};

export function Glyph({id, label, small}: GlyphProps): ReactNode {
  return (
    <svg
      className={small ? 'ms-glyphs ms-glyph ms-glyph--small' : 'ms-glyphs ms-glyph'}
      viewBox="0 0 24 24"
      role="img"
      aria-label={label}>
      <use href={`#ms-${id}`} />
    </svg>
  );
}

export function GlyphSprite({sprite}: {sprite: string}): ReactNode {
  return <div className="ms-sprite" aria-hidden="true" dangerouslySetInnerHTML={{__html: sprite}} />;
}
