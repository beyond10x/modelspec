import type {ReactNode} from 'react';
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
};

export default function ModelDiagram({data, view, scope, caption}: ModelDiagramProps): ReactNode {
  const drawn = data.views[view];
  const key = scope ?? drawn?.root;
  const standalone = useBaseUrl(`/img/diagrams/${data.program}/${view}/${key}.svg`);
  const svg = key === undefined ? undefined : drawn?.levels[key];
  if (svg === undefined) {
    throw new Error(`${data.program} has no level ${key ?? '(root)'} in its ${view} view`);
  }
  return (
    <figure className="ms-figure">
      <div className="ms-figure__canvas" dangerouslySetInnerHTML={{__html: svg}} />
      <figcaption>
        {caption ?? (
          <>
            The {view} view of <code>{key}</code>, drawn by <code>modelspec-diagram</code> from its
            verified Tensor IR.
          </>
        )}{' '}
        <a href={standalone}>Open the SVG</a>.
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
