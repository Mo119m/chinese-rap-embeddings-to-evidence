'use client';

import { FormEvent, useMemo, useRef, useState } from 'react';
import rawData from './data/researchData.json';
import rawCharacterMap from './data/characterToRhymeFamily.json';

type View = 'home' | 'repertoire' | 'references' | 'rhyme';
type Term = { text: string; score: number; stability: number; supportSongs: number };
type Trait = { key: 'short' | 'repeat' | 'mix'; percentile: number; raw: number };
type RhymeItem = { value: string; count: number; share: number };
type RhymeProfile = {
  dominantFamily: string;
  dominantShare: number;
  topFamilies: RhymeItem[];
  distinctiveFamilies: { family: string; log2_rate_ratio_vs_corpus: number }[];
  adjacentSameFamilyRate: number;
  echoLift: number;
  medianRun: number;
  lines: number;
  songs: number;
} | null;
type LabelNode = {
  id: string;
  label: string;
  x: number;
  y: number;
  independentSongs: number;
  terms: Term[];
  traits: Trait[];
  rhyme: RhymeProfile;
};
type LyricalReason = { kind: string; label: string; items: string[]; percentile: number };
type LyricalEdge = {
  a: string;
  b: string;
  repeatability: number;
  status: 'repeatable' | 'exploratory';
  dominantSignal: string;
  reasons: LyricalReason[];
};
type Entity = {
  id: string;
  name: string;
  type: string;
  songs: number;
  labels: number;
  agreementRate: number;
  status: string;
};
type ReferenceLink = {
  labelId: string;
  entityId: string;
  entityType: string;
  songs: number;
  labelSongs: number;
  share: number;
  lift: number;
  liftLow: number;
  liftHigh: number;
  qValue: number;
  agreementOccurrences: number;
  reliability: string;
  plainMeaning: string;
  status: string;
};
type CoMention = {
  a: string;
  aType: string;
  b: string;
  bType: string;
  songUnits: number;
  labels: number;
  lift: number;
  npmi: number;
  qValue: number;
  reliability: string;
};
type Recommendation = { written_rhyme_family: string; probability: number };
type RhymeLabelEvidence = {
  labelId: string;
  count: number;
  share: number;
  log2RateRatio: number;
  songs: number;
  lines: number;
};
type RhymeContext = {
  labelId: string;
  previous2: string;
  previous1: string;
  run: string;
  position: string;
  support: number;
  top5: Recommendation[];
};
type ResearchData = {
  question: string;
  constructDefinition: string;
  labels: LabelNode[];
  lyricalEdges: LyricalEdge[];
  repertoireGraph?: {
    representation: string;
    eligibleLabels: number;
    connectedLabels: number;
    retainedEdges: number;
    repeatableEdges: number;
    bootstrapReplicates: number;
    repeatabilityGate: number;
    pcaVariance2d: number;
    alignmentNull: {
      observed_intersection_edges: number;
      primary_edges: number;
      sensitivity_edges: number;
      null_replicates: number;
      null_mean: number;
      null_95_interval: number[];
      monte_carlo_p_add_one: number;
      estimand: string;
      null_model: string;
    };
    projectionFidelity: {
      pairwise_rank_spearman: number;
      neighbourhood_fidelity: {
        k: number;
        trustworthiness: number;
        mean_exact_neighbour_overlap: number;
        random_overlap_expectation: number;
      }[];
      released_edges_mutual_top5_in_2d: number;
      released_edges_at_least_one_way_top5_in_2d: number;
      interpretation: string;
    };
    edgeRule: string;
    layoutMeaning: string;
    claimBoundary: string;
  };
  retrieval: {
    systems: {
      name: string;
      mrr: { estimate: number; ci95_lower: number; ci95_upper: number };
      recall1: { estimate: number };
      recall5: { estimate: number };
      recall10: { estimate: number };
      ndcg10: { estimate: number };
    }[];
    queries: number;
    labels: number;
    groups: number;
    fusionMrrCi: number[];
    inductiveSanity: {
      folds: number;
      queries: number;
      inductiveTfidfMrr: number;
      inductiveFusionMrr: number;
      inductiveFusionGainMrr: { estimate_delta: number; ci95_lower: number; ci95_upper: number };
      tfidfExposureEffectMrr: { estimate_delta: number; ci95_lower: number; ci95_upper: number };
      fusionExposureEffectMrr: { estimate_delta: number; ci95_lower: number; ci95_upper: number };
      interpretation: string;
      claimBoundary: string;
    };
    claimBoundary: string;
  };
  ner: {
    entities: Entity[];
    links: ReferenceLink[];
    coMentions?: CoMention[];
    status: string;
    humanGoldAvailable: boolean;
    releasedClaimAudit: {
      status: string;
      releasedClaims: number;
      uniqueSupportingOccurrences: number;
      coverage: number;
      occurrenceDecisionsCompleted: number;
      coMentionDecisionsCompleted: number;
      evidenceBoundary: string;
      nextAction: string;
    };
    counts: Record<string, number>;
    claimBoundary: string;
  };
  rhyme: {
    classes: { written_rhyme_family: string; included_strict_pypinyin_finals: string[]; plain_description: string }[];
    familyOrder: string[];
    globalTop5: Recommendation[];
    markov: Record<string, { training_event_support: number; top_5: Recommendation[] }>;
    contexts: RhymeContext[];
    labelsByFamily: Record<string, RhymeLabelEvidence[]>;
    releasedModel: { name: string; selected_alpha: number; selected_validation_temperature: number };
    sourceLabelConditioning: {
      claim_policy: string;
      top3_and_mrr_intervals_support_positive_increment: boolean;
    };
    abstention: {
      recommended: { threshold: number };
      operating_points: {
        operating_point: string;
        test_coverage: number;
        top1_accuracy_on_accepted: number;
        top3_accuracy_on_accepted: number;
        mrr_on_accepted: number;
      }[];
    };
    metrics: {
      model: string;
      top1_accuracy: number;
      top3_accuracy: number;
      top5_accuracy: number;
      mrr: number;
    }[];
    switchDiagnostic: {
      stratum_value: string;
      top1_accuracy: string;
      top3_accuracy: string;
    }[];
    testEvents: number;
    testSongs: number;
    strictCandidateCoverage: number;
    claimBoundary: string;
  };
  publicBoundary: string;
};

const data = rawData as ResearchData;
const characterMap = rawCharacterMap as Record<string, string>;

const FAMILY_EXAMPLES: Record<string, string[]> = {
  A: ['家', '花', '大', '下'], O: ['我', '说', '过', '火'], E: ['的', '这', '歌', '河'],
  IE_VE: ['夜', '月', '写', '街'], AI: ['爱', '来', '海', '在'], EI: ['飞', '泪', '美', '回'],
  AO: ['到', '高', '好', '跑'], OU: ['走', '后', '手', '有'], AN: ['看', '山', '难', '安'],
  EN: ['人', '真', '门', '深'], ANG: ['光', '想', '上', '长'], ENG: ['梦', '城', '声', '等'],
  ONG: ['中', '风', '同', '空'], I: ['你', '里', '自', '是'], U: ['路', '苦', '住', '不'],
  V: ['去', '句', '绿', '雨'], ER: ['二', '耳', '儿', '尔'],
};

const SIGNAL_LABEL: Record<string, string> = {
  language: 'distinctive wording', lineEnding: 'written endings', form: 'writing habit', semanticOnly: 'overall repertoire',
};
const SIGNAL_COLOR: Record<string, string> = {
  language: '#8f7cff', lineEnding: '#ffb13b', form: '#25c7aa', semanticOnly: '#8e96a3',
};
const ENTITY_COLOR: Record<string, string> = {
  PLACE: '#246bfd', PERSON_REFERENCE: '#ff6846', LANGUAGE_OR_DIALECT_REFERENCE: '#9c6cff',
};
const ENTITY_LABEL: Record<string, string> = {
  PLACE: 'Place', PERSON_REFERENCE: 'Person reference', LANGUAGE_OR_DIALECT_REFERENCE: 'Language / dialect',
};

const pct = (value: number, digits = 0) => `${(value * 100).toFixed(digits)}%`;
const pp = (value: number, digits = 1) => `${(value * 100).toFixed(digits)} pp`;
const qText = (value: number) => value < .001 ? value.toExponential(1) : value.toFixed(3);
const labelById = new Map(data.labels.map((label) => [label.id, label]));
const entityById = new Map(data.ner.entities.map((entity) => [entity.id, entity]));
const graphMeta = data.repertoireGraph ?? {
  representation: 'BGE-M3 source-label centroids under two duplicate-controlled text treatments',
  eligibleLabels: data.labels.length,
  connectedLabels: new Set(data.lyricalEdges.flatMap((edge) => [edge.a, edge.b])).size,
  retainedEdges: data.lyricalEdges.length,
  repeatableEdges: data.lyricalEdges.filter((edge) => edge.status === 'repeatable').length,
  bootstrapReplicates: 250,
  repeatabilityGate: .5,
  pcaVariance2d: .261959589,
  alignmentNull: {
    observed_intersection_edges: 86,
    primary_edges: 140,
    sensitivity_edges: 145,
    null_replicates: 10000,
    null_mean: 4.5243,
    null_95_interval: [1, 9],
    monte_carlo_p_add_one: 1 / 10001,
    estimand: 'specific cross-treatment adjacency agreement after controlling the complete sensitivity-layer degree sequence',
    null_model: 'degree-preserving double-edge swaps of the sensitivity layer',
  },
  projectionFidelity: {
    pairwise_rank_spearman: .68028019,
    neighbourhood_fidelity: [{ k: 5, trustworthiness: .78488896, mean_exact_neighbour_overlap: .17254902, random_overlap_expectation: .02463054 }],
    released_edges_mutual_top5_in_2d: 18,
    released_edges_at_least_one_way_top5_in_2d: 28,
    interpretation: 'Use PCA for broad navigation. Read exact relationships from released lines, not screen distance.',
  },
  edgeRule: 'Both source-label profiles rank each other among their five closest matches in both text treatments.',
  layoutMeaning: 'Position is an approximate two-dimensional summary of BGE-M3 repertoire profiles; only a line defines a released match.',
  claimBoundary: 'Textual-repertoire proximity inside this corpus—not friendship, collaboration, influence, genre, geography, biography, preference, popularity, or verified identity.',
};

function traitText(trait: Trait): string {
  const high = trait.percentile >= 70;
  const low = trait.percentile <= 30;
  if (trait.key === 'short') return high ? 'mostly shorter written lines' : low ? 'mostly longer written lines' : 'mixed line lengths';
  if (trait.key === 'repeat') return high ? 'frequent exact-line reuse' : low ? 'rare exact-line reuse' : 'moderate line reuse';
  return high ? 'frequent Chinese–English mixing' : low ? 'rare Chinese–English mixing' : 'moderate Chinese–English mixing';
}

function reasonText(edge: LyricalEdge): string {
  const first = edge.reasons[0];
  if (!first) return 'No vocabulary, written-ending, or writing-form probe reached the preset display gate; the BGE-M3 profiles still satisfy the reciprocal match rule.';
  if (first.kind === 'language') return `Shared distinctive wording: ${first.items.join(' · ')}`;
  if (first.kind === 'lineEnding') return `Both favour these written endings: ${first.items.join(' · ')}`;
  if (first.kind === 'form') {
    const readable = first.items.map((item) => ({
      'high Chinese-English mixing': 'frequent Chinese–English mixing',
      'low Chinese-English mixing': 'rare Chinese–English mixing',
      'high repeated-line use': 'frequent line repetition',
      'low repeated-line use': 'rare line repetition',
      'high short-line writing': 'mostly shorter lines',
      'low short-line writing': 'mostly longer lines',
    }[item] ?? item));
    return `Shared writing habit: ${readable.join(' · ')}`;
  }
  return first.label;
}

function bestConnectedLabel(): LabelNode {
  const degree = new Map<string, number>();
  data.lyricalEdges.filter((edge) => edge.status === 'repeatable').forEach((edge) => {
    degree.set(edge.a, (degree.get(edge.a) ?? 0) + 1);
    degree.set(edge.b, (degree.get(edge.b) ?? 0) + 1);
  });
  return [...data.labels].sort((a, b) => (degree.get(b.id) ?? 0) - (degree.get(a.id) ?? 0))[0];
}

function bestReferencedLabel(): LabelNode {
  const score = new Map<string, number>();
  data.ner.links.forEach((link) => score.set(link.labelId, (score.get(link.labelId) ?? 0) + link.songs));
  return [...data.labels].sort((a, b) => (score.get(b.id) ?? 0) - (score.get(a.id) ?? 0))[0];
}

function LabelSearch({ selected, onSelect, available = data.labels }: { selected: LabelNode; onSelect: (label: LabelNode) => void; available?: LabelNode[] }) {
  const [query, setQuery] = useState(selected.label);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    const normal = query.trim().toLocaleLowerCase();
    const match = available.find((label) => label.label.toLocaleLowerCase() === normal)
      ?? available.find((label) => label.label.toLocaleLowerCase().includes(normal));
    if (match) { onSelect(match); setQuery(match.label); }
  };
  return (
    <form className="label-search" onSubmit={submit}>
      <label htmlFor={`label-${selected.id}`}>Source-credit label</label>
      <div className="input-row">
        <input id={`label-${selected.id}`} list={`labels-${selected.id}`} value={query} onChange={(event) => setQuery(event.target.value)} />
        <datalist id={`labels-${selected.id}`}>{available.map((label) => <option key={label.id} value={label.label} />)}</datalist>
        <button type="submit">Open</button>
      </div>
    </form>
  );
}

function Header({ view, setView }: { view: View; setView: (view: View) => void }) {
  return (
    <header className="site-header">
      <button className="wordmark" onClick={() => setView('home')} aria-label="Verseprint home">
        <span className="mark">V//P</span><span><b>VERSEPRINT</b><small>Chinese Rap Evidence Lab</small></span>
      </button>
      <nav aria-label="Primary">
        {([['repertoire', 'Repertoire map'], ['references', 'Reference network'], ['rhyme', 'Rhyme lab']] as [View, string][]).map(([key, text]) => (
          <button key={key} className={view === key ? 'active' : ''} onClick={() => setView(key)}>{text}</button>
        ))}
      </nav>
      <span className="release-chip">RESEARCH BUILD 01</span>
    </header>
  );
}

function Home({ setView }: { setView: (view: View) => void }) {
  const cards: { key: View; number: string; title: string; action: string; description: string; proof: string }[] = [
    { key: 'repertoire', number: '01', title: 'Who writes nearby?', action: 'Open repertoire map', description: 'Start with all 204 eligible source labels, then select one to inspect its retained repertoire neighbours and measurable common ground.', proof: '86 cross-treatment matches versus 4.52 under degree-preserving random rewires.' },
    { key: 'references', number: '02', title: 'Which cultural worlds appear?', action: 'Open reference network', description: 'Move from one source label to a supported lyric reference, then see which other labels invoke the same reference.', proof: 'All 157 occurrences behind the 10 released claims are queued for blind dual review; none is yet adjudicated.' },
    { key: 'rhyme', number: '03', title: 'What ending could come next?', action: 'Open rhyme lab', description: 'Enter a Chinese ending to receive ranked next written-ending families, then see which repertoires emphasize a selected family.', proof: 'Sequential context is tested on song-held-out data; label personalization is not claimed.' },
  ];
  return (
    <main className="home-shell">
      <section className="hero"><p className="eyebrow">ONE CORPUS · THREE EVIDENCE-GRADED TASKS</p><h1>What makes a Chinese rap<br /><em>lyrical identity</em> recognizable?</h1><p className="hero-question">{data.constructDefinition}</p></section>
      <section className="task-grid" aria-label="Result tools">
        {cards.map((card) => <button key={card.key} className={`task-card task-${card.number}`} onClick={() => setView(card.key)}><span className="task-number">{card.number}</span><h2>{card.title}</h2><p>{card.description}</p><strong>{card.proof}</strong><span className="card-action">{card.action} <i>↗</i></span></button>)}
      </section>
      <footer className="home-foot"><span>TEXT-DERIVED · SONG-HELD-OUT · DUPLICATE-CONTROLLED</span><span>No lyric lines are published.</span></footer>
    </main>
  );
}

function GlobalRepertoireGraph({ selected, onSelect }: { selected: LabelNode; onSelect: (node: LabelNode) => void }) {
  const [mode, setMode] = useState<'all' | 'repeatable'>('all');
  const [display, setDisplay] = useState<'map' | 'table'>('map');
  const visibleEdges = useMemo(() => mode === 'repeatable' ? data.lyricalEdges.filter((edge) => edge.status === 'repeatable') : data.lyricalEdges, [mode]);
  const linkedIds = useMemo(() => new Set(visibleEdges.flatMap((edge) => [edge.a, edge.b])), [visibleEdges]);
  const neighbourRows = useMemo(() => {
    const all = new Map<string, string[]>();
    const repeatable = new Map<string, string[]>();
    data.lyricalEdges.forEach((edge) => {
      const a = labelById.get(edge.a)?.label ?? edge.a;
      const b = labelById.get(edge.b)?.label ?? edge.b;
      all.set(edge.a, [...(all.get(edge.a) ?? []), b]);
      all.set(edge.b, [...(all.get(edge.b) ?? []), a]);
      if (edge.status === 'repeatable') {
        repeatable.set(edge.a, [...(repeatable.get(edge.a) ?? []), b]);
        repeatable.set(edge.b, [...(repeatable.get(edge.b) ?? []), a]);
      }
    });
    return [...data.labels]
      .sort((a, b) => a.label.localeCompare(b.label, 'zh-CN'))
      .map((node) => ({ node, all: all.get(node.id) ?? [], repeatable: repeatable.get(node.id) ?? [] }));
  }, []);
  const repeatableDegree = useMemo(() => {
    const result = new Map<string, number>();
    data.lyricalEdges.filter((edge) => edge.status === 'repeatable').forEach((edge) => {
      result.set(edge.a, (result.get(edge.a) ?? 0) + 1);
      result.set(edge.b, (result.get(edge.b) ?? 0) + 1);
    });
    return result;
  }, []);
  const point = (node: LabelNode) => ({ x: 64 + ((node.x + 1) / 2) * 1072, y: 42 + (1 - ((node.y + 1) / 2)) * 496 });
  const labelled = new Set([...data.labels].sort((a, b) => (repeatableDegree.get(b.id) ?? 0) - (repeatableDegree.get(a.id) ?? 0)).slice(0, 8).map((node) => node.id));
  labelled.add(selected.id);
  const fidelity5 = graphMeta.projectionFidelity.neighbourhood_fidelity.find((row) => row.k === 5)!;
  return (
    <section className="global-network" aria-labelledby="global-network-title">
      <div className="global-network-head">
        <div><p className="micro-label">CORPUS OVERVIEW</p><h2 id="global-network-title">The full repertoire landscape</h2><p>See the whole corpus first. Click any dot to carry that label into the focused view below.</p></div>
        <div className="global-controls"><div className="network-toggle" aria-label="Visible repertoire edges"><button className={mode === 'all' ? 'active' : ''} onClick={() => setMode('all')}>All {graphMeta.retainedEdges} released</button><button className={mode === 'repeatable' ? 'active' : ''} onClick={() => setMode('repeatable')}>{graphMeta.repeatableEdges} display-gate matches</button></div><div className="view-toggle" aria-label="Overview format"><button className={display === 'map' ? 'active' : ''} onClick={() => setDisplay('map')}>Map</button><button className={display === 'table' ? 'active' : ''} onClick={() => setDisplay('table')}>Accessible table</button></div></div>
      </div>
      <div className="global-counts"><div><b>{graphMeta.eligibleLabels}</b><span>eligible labels</span></div><div><b>{graphMeta.connectedLabels}</b><span>with a released line</span></div><div><b>{graphMeta.retainedEdges}</b><span>reciprocal matches</span></div><div><b>{graphMeta.repeatableEdges}</b><span>returned in ≥50% of repeats</span></div></div>
      <div className="robustness-proof" aria-label="Network and projection validation"><div><span>GRAPH AGREEMENT</span><b>{graphMeta.alignmentNull.observed_intersection_edges} observed vs {graphMeta.alignmentNull.null_mean.toFixed(2)} random</b><p>None of {graphMeta.alignmentNull.null_replicates.toLocaleString()} degree-preserving rewires matched the observed overlap (p≈{graphMeta.alignmentNull.monte_carlo_p_add_one.toExponential(1)}).</p></div><div><span>MAP FIDELITY</span><b>{pct(fidelity5.trustworthiness, 1)} trustworthiness@5</b><p>{pct(fidelity5.mean_exact_neighbour_overlap, 1)} of exact high-dimensional top-five neighbours remain top-five on the page.</p></div><div><span>READING RULE</span><b>Use lines, not screen distance</b><p>{graphMeta.projectionFidelity.interpretation}</p></div></div>
      {display === 'map' ? <>
        <div className="global-canvas">
          <svg viewBox="0 0 1200 580" aria-hidden="true" focusable="false">
            {visibleEdges.map((edge) => { const a = labelById.get(edge.a); const b = labelById.get(edge.b); if (!a || !b) return null; const pa = point(a); const pb = point(b); const color = SIGNAL_COLOR[edge.dominantSignal] ?? SIGNAL_COLOR.semanticOnly; return <line key={`${edge.a}-${edge.b}`} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y} stroke={color} strokeWidth={edge.status === 'repeatable' ? 1.6 + edge.repeatability * 2.2 : .8 + edge.repeatability} strokeOpacity={edge.status === 'repeatable' ? .78 : .20} strokeDasharray={edge.status === 'exploratory' ? '3 5' : undefined}><title>{`${a.label} ↔ ${b.label}; returned in ${Math.round(edge.repeatability * graphMeta.bootstrapReplicates)} of ${graphMeta.bootstrapReplicates} repeated song samples`}</title></line>; })}
            {data.labels.map((node) => { const p = point(node); const active = node.id === selected.id; const connected = linkedIds.has(node.id); const radius = 3.8 + Math.min(7.5, Math.sqrt(Math.max(node.independentSongs, 1)) * .7); return <g key={node.id} className={`overview-node ${active ? 'selected' : ''}`} transform={`translate(${p.x} ${p.y})`} onClick={() => onSelect(node)}><circle r={active ? radius + 4 : radius} fill={active ? '#e5ff3d' : connected ? '#f8f5ed' : '#69717d'} fillOpacity={connected || active ? .96 : .38} stroke={active ? '#11141a' : '#f8f5ed'} strokeWidth={active ? 2.5 : .7}><title>{`${node.label}: ${node.independentSongs} independent-song support${connected ? '' : '; no line released under the current rule'}`}</title></circle>{labelled.has(node.id) && <text x={radius + 6} y="4">{node.label}</text>}</g>; })}
          </svg>
        </div>
        <div className="global-legend"><span><i className="node-key" />Dot size = independent-song support, not popularity</span>{Object.entries(SIGNAL_LABEL).map(([key, label]) => <span key={key}><i style={{ background: SIGNAL_COLOR[key] }} />{label}</span>)}<span><b>solid</b> ≥50% display gate</span><span><b>dashed</b> lower-repeatability candidate</span></div>
      </> : <div className="graph-table-wrap"><table><caption>All {graphMeta.eligibleLabels} source-credit labels represented in the overview</caption><thead><tr><th scope="col">Source-credit label</th><th scope="col">Independent-song support</th><th scope="col">Released neighbours</th><th scope="col">≥50% repeatable</th></tr></thead><tbody>{neighbourRows.map((row) => <tr key={row.node.id}><th scope="row"><button onClick={() => onSelect(row.node)}>{row.node.label}</button></th><td>{row.node.independentSongs}</td><td>{row.all.length ? row.all.join(' · ') : 'No released line'}</td><td>{row.repeatable.length ? row.repeatable.join(' · ') : 'None'}</td></tr>)}</tbody></table></div>}
      <p className="map-boundary"><b>How to read it:</b> {graphMeta.layoutMeaning} The two axes retain {pct(graphMeta.pcaVariance2d, 1)} of profile variation. {graphMeta.claimBoundary}</p>
    </section>
  );
}

function RepertoireGraph({ selected, onSelect, selectedEdge, onEdge }: { selected: LabelNode; onSelect: (node: LabelNode) => void; selectedEdge: LyricalEdge | null; onEdge: (edge: LyricalEdge) => void }) {
  const adjacent = useMemo(() => data.lyricalEdges.filter((edge) => edge.a === selected.id || edge.b === selected.id).sort((a, b) => (a.status === b.status ? b.repeatability - a.repeatability : a.status === 'repeatable' ? -1 : 1)).slice(0, 7), [selected.id]);
  const center = { x: 380, y: 278 };
  const points = adjacent.map((edge, index) => {
    const angle = -Math.PI / 2 + index * (Math.PI * 2 / Math.max(adjacent.length, 1));
    const distance = 205;
    const other = labelById.get(edge.a === selected.id ? edge.b : edge.a)!;
    return { edge, other, x: center.x + Math.cos(angle) * distance, y: center.y + Math.sin(angle) * distance };
  });
  return (
    <div className="graph-stage">
      <svg viewBox="0 0 760 556" role="img" aria-label={`Closest lyrical repertoires around ${selected.label}`}>
        <defs><filter id="soft"><feDropShadow dx="0" dy="8" stdDeviation="10" floodOpacity="0.24" /></filter></defs>
        {points.map(({ edge, other, x, y }) => { const active = selectedEdge === edge; const color = SIGNAL_COLOR[edge.dominantSignal] ?? SIGNAL_COLOR.semanticOnly; const mx = (x + center.x) / 2; const my = (y + center.y) / 2; const returns = Math.round(edge.repeatability * graphMeta.bootstrapReplicates); return <g key={`${edge.a}-${edge.b}`} className={`graph-edge ${active ? 'active' : ''}`} role="button" tabIndex={0} aria-pressed={active} aria-label={`Explain match to ${other.label}: ${reasonText(edge)}; returned in ${returns} of ${graphMeta.bootstrapReplicates} repeated song samples`} onClick={() => onEdge(edge)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onEdge(edge); } }}><circle cx={mx} cy={my} r="22" fill="transparent" /><line x1={center.x} y1={center.y} x2={x} y2={y} stroke={color} strokeWidth={active ? 6 : 2 + edge.repeatability * 3} strokeDasharray={edge.status === 'exploratory' ? '5 8' : undefined} /><circle cx={mx} cy={my} r="15" fill="#11141a" stroke={color} /><text x={mx} y={my + 4} textAnchor="middle" className="edge-percent">{returns}</text><title>{`${other.label}: returned in ${returns} of ${graphMeta.bootstrapReplicates} repeated song samples`}</title></g>; })}
        {points.map(({ edge, other, x, y }) => <g key={other.id} className="graph-node" transform={`translate(${x} ${y})`} role="button" tabIndex={0} aria-label={`Select ${other.label} repertoire`} onClick={() => onSelect(other)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(other); } }}><circle r={edge.status === 'repeatable' ? 28 : 23} fill={edge.status === 'repeatable' ? '#f4f0e8' : '#20262f'} stroke={SIGNAL_COLOR[edge.dominantSignal] ?? '#8e96a3'} strokeWidth="3" /><text y={edge.status === 'repeatable' ? 44 : 39} textAnchor="middle" className="node-name">{other.label}</text></g>)}
        <g className="center-node" transform={`translate(${center.x} ${center.y})`} filter="url(#soft)"><circle r="72" /><text y="-5" textAnchor="middle">{selected.label}</text><text y="20" textAnchor="middle" className="center-sub">selected repertoire</text></g>
        {adjacent.length === 0 && <text x="380" y="395" textAnchor="middle" className="empty-svg">No reciprocal match is released for this label under the current rule.</text>}
      </svg>
      <div className="graph-legend">{Object.entries(SIGNAL_LABEL).map(([key, label]) => <span key={key}><i style={{ background: SIGNAL_COLOR[key] }} />{label}</span>)}<span><b>number</b> returns out of 250</span><span><b>solid</b> display gate met</span><span><b>dashed</b> lower-repeatability</span></div>
    </div>
  );
}

function RepertoireView() {
  const [selected, setSelected] = useState<LabelNode>(bestConnectedLabel());
  const [selectedEdge, setSelectedEdge] = useState<LyricalEdge | null>(null);
  const localRef = useRef<HTMLElement | null>(null);
  const adjacent = useMemo(() => data.lyricalEdges.filter((edge) => edge.a === selected.id || edge.b === selected.id).sort((a, b) => (a.status === b.status ? b.repeatability - a.repeatability : a.status === 'repeatable' ? -1 : 1)), [selected.id]);
  const standout = selected.traits.length ? [...selected.traits].sort((a, b) => Math.abs(b.percentile - 50) - Math.abs(a.percentile - 50))[0] : null;
  const distinctiveEnding = selected.rhyme?.distinctiveFamilies?.[0] ?? null;
  const inductive = data.retrieval.inductiveSanity;
  const openLabel = (label: LabelNode) => { setSelected(label); setSelectedEdge(null); };
  const openFromOverview = (label: LabelNode) => { openLabel(label); requestAnimationFrame(() => localRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })); };
  return (
    <main className="tool-shell">
      <section className="tool-intro"><div><p className="eyebrow">RESULT 01 · LANGUAGE</p><h1>Lyrical repertoire map</h1><p>Start with the full 204-label landscape, then open one repertoire to see its released neighbours, writing signature, and the evidence behind every line.</p></div><LabelSearch selected={selected} onSelect={openLabel} available={data.labels} /></section>
      <GlobalRepertoireGraph selected={selected} onSelect={openFromOverview} />
      <section className="local-network-head" ref={localRef}><p className="micro-label">FOCUSED VIEW</p><h2>{selected.label} and its retained neighbours</h2><p>This circle is an ego diagram for navigation; angle and distance carry no additional meaning. Line colour shows an auxiliary writing signal, while the number shows how often the same match returned across 250 repeated song samples.</p></section>
      <section className="workspace two-column"><RepertoireGraph selected={selected} onSelect={openLabel} selectedEdge={selectedEdge} onEdge={setSelectedEdge} /><aside className="result-panel">
        <div className="panel-head"><span className="status-dot verified" /><span>{selected.independentSongs} independent-song profile</span></div><h2>{selected.label}</h2>
        <p className="lead-result">{standout ? `Most distinctive writing habit: ${traitText(standout)}.` : 'This label has reference evidence but not enough material for a full repertoire profile.'} {distinctiveEnding ? `Ending signature: ${distinctiveEnding.family} appears ${Math.pow(2, distinctiveEnding.log2_rate_ratio_vs_corpus).toFixed(1)}× as often as in the corpus overall (for example ${FAMILY_EXAMPLES[distinctiveEnding.family]?.join(' / ') ?? distinctiveEnding.family}).` : ''}</p>
        {selected.terms.length > 0 && <section className="panel-section"><h3>Distinctive wording</h3><div className="term-cloud">{selected.terms.map((term) => <span key={term.text} title={`Supported across ${term.supportSongs} songs`}>{term.text}</span>)}</div></section>}
        {selected.rhyme && <section className="panel-section"><h3>Written-ending fingerprint</h3><div className="family-row">{selected.rhyme.topFamilies.slice(0, 4).map((item) => <div key={item.value}><b>{item.value}</b><span>{pct(item.share)}</span></div>)}</div><p className="meaning-line">Adjacent lines retain the same ending family {pct(selected.rhyme.adjacentSameFamilyRate)} of the time.</p></section>}
        <section className="panel-section"><h3>{selectedEdge ? 'What supports this match' : 'Retained repertoire matches'}</h3>{selectedEdge ? <div className="why-card" style={{ borderColor: SIGNAL_COLOR[selectedEdge.dominantSignal] }}><b>{labelById.get(selectedEdge.a === selected.id ? selectedEdge.b : selectedEdge.a)?.label}</b><p><strong>Why there is a line:</strong> {graphMeta.edgeRule}</p><p><strong>Also shared:</strong> {reasonText(selectedEdge)}</p><span className={selectedEdge.status === 'repeatable' ? 'evidence strong' : 'evidence provisional'}>Returned in {Math.round(selectedEdge.repeatability * graphMeta.bootstrapReplicates)}/{graphMeta.bootstrapReplicates} repeated song samples · {selectedEdge.status === 'repeatable' ? 'display gate met' : 'lower-repeatability candidate'}</span><button onClick={() => setSelectedEdge(null)}>Back to all matches</button></div> : adjacent.slice(0, 5).map((edge) => { const other = labelById.get(edge.a === selected.id ? edge.b : edge.a)!; return <button className="match-row" key={`${edge.a}-${edge.b}`} onClick={() => setSelectedEdge(edge)}><span className="signal-bar" style={{ background: SIGNAL_COLOR[edge.dominantSignal] }} /><span><b>{other.label}</b><small>{reasonText(edge)}</small></span><em>{Math.round(edge.repeatability * graphMeta.bootstrapReplicates)}</em></button>; })}</section>
      </aside></section>
      <section className="evidence-strip"><div><span>What builds this map</span><b>BGE-M3 source-label profiles only; the network is descriptive and separate from the held-out retrieval benchmark.</b></div><div><span>Six-fold leakage check</span><b>With TF-IDF fitted on training folds only, fusion reaches {inductive.inductiveFusionMrr.toFixed(3)} vs {inductive.inductiveTfidfMrr.toFixed(3)} MRR (+{inductive.inductiveFusionGainMrr.estimate_delta.toFixed(3)}; 95% CI +{inductive.inductiveFusionGainMrr.ci95_lower.toFixed(3)}–+{inductive.inductiveFusionGainMrr.ci95_upper.toFixed(3)}). Test-distribution exposure adds only +{inductive.fusionExposureEffectMrr.estimate_delta.toFixed(3)}.</b></div><div><span>Boundary</span><b>{graphMeta.claimBoundary}</b></div></section>
    </main>
  );
}

function ReferenceGraph({ selected, selectedEntity, onEntity }: { selected: LabelNode; selectedEntity: Entity | null; onEntity: (entity: Entity) => void }) {
  const links = data.ner.links.filter((link) => link.labelId === selected.id).sort((a, b) => b.songs * Math.log2(b.lift + 1) - a.songs * Math.log2(a.lift + 1)).slice(0, 7);
  const entityPoints = links.map((link, index) => ({ link, entity: entityById.get(link.entityId)!, x: 375, y: 78 + index * (400 / Math.max(links.length - 1, 1)) }));
  const shared = selectedEntity ? data.ner.links.filter((link) => link.entityId === selectedEntity.id && link.labelId !== selected.id).sort((a, b) => b.lift * b.songs - a.lift * a.songs).slice(0, 6) : [];
  return (
    <div className="graph-stage reference-stage"><svg viewBox="0 0 760 556" role="img" aria-label={`Cultural references in lyrics credited to ${selected.label}`}>
      <text x="100" y="36" className="column-label">SOURCE LABEL</text><text x="334" y="36" className="column-label">LYRIC REFERENCE</text><text x="626" y="36" className="column-label">ALSO REFERENCES</text>
      {entityPoints.map(({ link, entity, x, y }) => <line key={`a-${entity.id}`} x1="145" y1="278" x2={x} y2={y} stroke={ENTITY_COLOR[entity.type] ?? '#999'} strokeWidth={1.5 + Math.min(5, link.songs / 3)} opacity={selectedEntity && selectedEntity.id !== entity.id ? .14 : .72} />)}
      {shared.map((link, index) => { const y = 120 + index * 62; const sourcePoint = entityPoints.find((point) => point.entity.id === selectedEntity?.id); return <line key={`b-${link.labelId}`} x1={sourcePoint?.x ?? 375} y1={sourcePoint?.y ?? 278} x2="650" y2={y} stroke={ENTITY_COLOR[selectedEntity?.type ?? ''] ?? '#999'} strokeWidth="2" opacity=".64" />; })}
      <g className="reference-center" transform="translate(145 278)"><circle r="70" /><text textAnchor="middle" y="-4">{selected.label}</text><text textAnchor="middle" y="18" className="center-sub">lyrics credited here</text></g>
      {entityPoints.map(({ entity, x, y }) => <g key={entity.id} className={`entity-node ${selectedEntity?.id === entity.id ? 'selected' : ''}`} transform={`translate(${x} ${y})`} role="button" tabIndex={0} aria-pressed={selectedEntity?.id === entity.id} aria-label={`Show evidence for ${entity.name}, ${ENTITY_LABEL[entity.type] ?? entity.type}`} onClick={() => onEntity(entity)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onEntity(entity); } }}><circle r="34" fill={ENTITY_COLOR[entity.type] ?? '#999'} /><text textAnchor="middle" y="5">{entity.name}</text></g>)}
      {shared.map((link, index) => { const label = labelById.get(link.labelId)!; return <g key={link.labelId} className="shared-label" transform={`translate(650 ${120 + index * 62})`}><circle r="19" /><text x="28" y="5">{label.label}</text></g>; })}
      {links.length === 0 && <text x="375" y="286" textAnchor="middle" className="empty-svg">No released reference survives the current evidence gate for this label.</text>}
    </svg><div className="graph-legend">{Object.entries(ENTITY_LABEL).map(([key, label]) => <span key={key}><i style={{ background: ENTITY_COLOR[key] }} />{label}</span>)}</div></div>
  );
}

function ReferenceView() {
  const audit = data.ner.releasedClaimAudit;
  const availableIds = new Set(data.ner.links.map((link) => link.labelId));
  const available = data.labels.filter((label) => availableIds.has(label.id));
  const [selected, setSelected] = useState<LabelNode>(bestReferencedLabel());
  const initialLink = data.ner.links.find((link) => link.labelId === selected.id);
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(initialLink ? entityById.get(initialLink.entityId)! : null);
  const openLabel = (label: LabelNode) => { setSelected(label); const first = data.ner.links.filter((link) => link.labelId === label.id).sort((a, b) => b.songs * b.lift - a.songs * a.lift)[0]; setSelectedEntity(first ? entityById.get(first.entityId)! : null); };
  const selectedLink = selectedEntity ? data.ner.links.find((link) => link.labelId === selected.id && link.entityId === selectedEntity.id) : null;
  const sharedLabels = selectedEntity ? data.ner.links.filter((link) => link.entityId === selectedEntity.id && link.labelId !== selected.id) : [];
  return (
    <main className="tool-shell"><section className="tool-intro"><div><p className="eyebrow">RESULT 02 · CULTURAL REFERENCE</p><h1>Reference network</h1><p>Select a label, then a place, language, or person reference. The next column shows who else invokes the same reference.</p></div><LabelSearch selected={selected} onSelect={openLabel} available={available} /></section>
      <section className="workspace two-column"><ReferenceGraph selected={selected} selectedEntity={selectedEntity} onEntity={setSelectedEntity} /><aside className="result-panel"><div className="panel-head"><span className="status-dot provisional" /><span>Claims queued for dual review · 0 adjudicated</span></div><h2>{selectedEntity?.name ?? selected.label}</h2>{selectedEntity && selectedLink ? <><p className="lead-result">After shared-text exclusion, <b>{selectedEntity.name}</b> appears in {selectedLink.songs} of {selectedLink.labelSongs} eligible <b>{selected.label}</b> song units ({pct(selectedLink.share, 1)}). Its shrunken enrichment is {selectedLink.lift.toFixed(1)}× versus other labels.</p><section className="panel-section"><h3>What the edge means</h3><div className="definition-card"><span>{ENTITY_LABEL[selectedEntity.type] ?? selectedEntity.type}</span><p>A repeated textual reference inside this source-labelled repertoire. It does not establish hometown, affiliation, belief, collaboration, preference, or a real-world relationship.</p></div></section><section className="panel-section"><h3>Who else references it?</h3>{sharedLabels.length ? <div className="shared-list">{sharedLabels.sort((a, b) => b.lift * b.songs - a.lift * a.songs).slice(0, 6).map((link) => <button key={link.labelId} onClick={() => openLabel(labelById.get(link.labelId)!)}><b>{labelById.get(link.labelId)?.label}</b><span>{link.songs}/{link.labelSongs} song units · {link.lift.toFixed(1)}× shrunken enrichment</span></button>)}</div> : <p className="meaning-line">No other source-label edge to this reference survives the same uncertainty and FDR gates.</p>}</section><section className="panel-section"><h3>Reliability</h3><p className="meaning-line">Conservative 95% enrichment interval: {selectedLink.liftLow.toFixed(1)}–{selectedLink.liftHigh.toFixed(1)}×; BH q={qText(selectedLink.qValue)}; class {selectedLink.reliability.toLocaleLowerCase()}. All {audit.uniqueSupportingOccurrences} unique occurrences behind the {audit.releasedClaims} released claims are in a blinded dual-review package, but no human adjudication is complete; precision and recall remain unknown.</p></section></> : <p className="lead-result">No entity clears shared-text, support, uncertainty, and FDR gates for this label.</p>}</aside></section>
      {!!data.ner.coMentions?.length && <section className="co-mention-wrap"><div className="co-mention-head"><p className="micro-label">SEPARATE RELATION TYPE</p><h2>References appearing in the same songs</h2><p>These four pairs co-occur more than expected across all 5,681 eligible song units after shared-text exclusion. They connect lyric references—not people or social groups.</p></div><div className="co-mention-grid">{data.ner.coMentions.map((pair) => <article key={`${pair.a}-${pair.b}`}><div><b>{pair.a}</b><i /><b>{pair.b}</b></div><strong>{pair.songUnits} song units across {pair.labels} source labels</strong><span>Positive association NPMI {pair.npmi.toFixed(2)} · BH q={qText(pair.qValue)}</span></article>)}</div></section>}
      <section className="evidence-strip"><div><span>Released statistical layer</span><b>{data.ner.links.length} label–reference edges and {data.ner.coMentions?.length ?? 0} co-mention pairs survive shared-text, support, uncertainty, and BH-FDR gates.</b></div><div><span>Human-audit readiness</span><b>{pct(audit.coverage)} of the {audit.uniqueSupportingOccurrences} supporting occurrences are queued for blind dual review; {audit.occurrenceDecisionsCompleted + audit.coMentionDecisionsCompleted} are adjudicated.</b></div><div><span>Boundary</span><b>Edges encode provisional lyric references—not social relations, biography, or preference.</b></div></section>
    </main>
  );
}

function metric(model: string) { return data.rhyme.metrics.find((row) => row.model === model)!; }

function RhymeView() {
  const [ending, setEnding] = useState('你');
  const [switchOnly, setSwitchOnly] = useState(false);
  const [focusedFamily, setFocusedFamily] = useState('');
  const han = [...ending].filter((character) => /[\u3400-\u9fff]/.test(character)).at(-1) ?? '';
  const family = characterMap[han] ?? '';
  const markov = family ? data.rhyme.markov[family] : undefined;
  const source = markov?.top_5 ?? data.rhyme.globalTop5;
  const recommendations = switchOnly && family ? source.filter((row) => row.written_rhyme_family !== family) : source;
  const evidenceFamily = focusedFamily || family;
  const relatedLabels = evidenceFamily ? (data.rhyme.labelsByFamily[evidenceFamily] ?? []).slice(0, 5) : [];
  const global = metric('global_frequency'); const markovMetric = metric('first_order_markov'); const modelMetric = metric('hierarchical_sgd_context');
  const selective = data.rhyme.abstention.operating_points.find((row) => row.operating_point === 'validation_target_50pct_coverage')!;
  const continueRow = data.rhyme.switchDiagnostic.find((row) => row.stratum_value === 'continuation');
  const changeRow = data.rhyme.switchDiagnostic.find((row) => row.stratum_value === 'switch');
  return (
    <main className="tool-shell rhyme-shell"><section className="tool-intro"><div><p className="eyebrow">RESULT 03 · WRITTEN RHYME</p><h1>Next-ending lab</h1><p>Enter a Chinese line ending. The tool ranks likely next written-ending families and shows which corpus repertoires emphasize the selected family.</p></div><div className="rhyme-controls"><label>Line ending<input value={ending} onChange={(event) => { setEnding(event.target.value); setFocusedFamily(''); }} maxLength={20} placeholder="e.g. 爱" /></label></div></section>
      <section className="rhyme-workspace"><div className="rhyme-input-card"><p className="micro-label">DICTIONARY ESTIMATE</p><div className="ending-readout"><span>{han || '—'}</span><div><b>{family || 'No ending recognized'}</b><small>{family ? `${FAMILY_EXAMPLES[family]?.join(' · ')} share this broad family` : 'Try one Chinese character or choose an example below.'}</small></div></div><div className="example-buttons">{['爱', '梦', '路', '光', '城', '你'].map((example) => <button key={example} onClick={() => { setEnding(example); setFocusedFamily(''); }}>{example}</button>)}</div><div className="toggle-row"><button className={!switchOnly ? 'active' : ''} onClick={() => setSwitchOnly(false)}>All options</button><button className={switchOnly ? 'active' : ''} onClick={() => setSwitchOnly(true)}>Switch options only</button></div><p className="boundary-note">Dictionary pinyin represents written text only. It cannot hear pronunciation, flow, cadence, or beat.</p></div>
        <div className="recommendation-card"><div className="recommendation-head"><div><p className="micro-label">INTERPRETABLE ONE-STEP TABLE</p><h2>{family ? `After ${family}` : 'Corpus-wide baseline'}</h2></div><span className="answer-state fallback">{family ? 'observed transition' : 'global baseline'}</span></div><div className="recommendations">{recommendations.slice(0, 5).map((item, index) => { const globalItem = data.rhyme.globalTop5.find((candidate) => candidate.written_rhyme_family === item.written_rhyme_family); const lift = globalItem ? item.probability / globalItem.probability : null; return <button type="button" className={`recommendation ${focusedFamily === item.written_rhyme_family ? 'selected' : ''}`} key={item.written_rhyme_family} onClick={() => setFocusedFamily(item.written_rhyme_family)}><span className="rank">0{index + 1}</span><span className="family"><b>{item.written_rhyme_family}</b><small>Try {FAMILY_EXAMPLES[item.written_rhyme_family]?.join(' / ')}</small></span><span className="probability"><b>{pct(item.probability, 1)}</b><span>{lift === null ? 'outside global top 5' : lift >= 1 ? `${lift.toFixed(1)}× global` : 'below global'}</span></span><span className="prob-bar"><i style={{ width: `${Math.min(100, item.probability * 180)}%` }} /></span></button>; })}</div><p className="recommendation-why">{family ? `This visible list is a smoothed one-step table learned from ${markov?.training_event_support.toLocaleString()} training events that originally followed ${family}. The higher-scoring full-context model is evaluated separately in the dark panel; it is not the list shown here.` : 'Enter an ending to use the one-step table.'}</p><section className="family-evidence"><h3>Repertoires emphasizing {evidenceFamily || 'this family'}</h3>{relatedLabels.length ? <div className="family-label-list">{relatedLabels.map((row) => <div key={row.labelId}><b>{labelById.get(row.labelId)?.label}</b><span>{pct(row.share, 1)} of written endings · {Math.pow(2, row.log2RateRatio).toFixed(1)}× corpus rate</span></div>)}</div> : <p>No label clears the public support gate for this family.</p>}<small>Descriptive corpus evidence only—not an intrinsic rapper preference.</small></section></div>
        <aside className="rhyme-proof"><p className="micro-label">HOW RELIABLE?</p><h2>Useful for ranked options.<br />Exact switches remain hard.</h2><div className="metric-stack"><div><span>Global Top-3</span><i><b style={{ width: `${global.top3_accuracy * 100}%` }} /></i><em>{pct(global.top3_accuracy, 1)}</em></div><div><span>One-step Top-3</span><i><b style={{ width: `${markovMetric.top3_accuracy * 100}%` }} /></i><em>{pct(markovMetric.top3_accuracy, 1)}</em></div><div className="highlight"><span>Full-context Top-3</span><i><b style={{ width: `${modelMetric.top3_accuracy * 100}%` }} /></i><em>{pct(modelMetric.top3_accuracy, 1)}</em></div></div><p className="proof-result">Full sequential context adds {pp(modelMetric.top3_accuracy - markovMetric.top3_accuracy)} over the one-step baseline on held-out songs.</p><div className="diagnostic"><span><b>{continueRow ? pct(Number(continueRow.top1_accuracy), 1) : '98.5%'}</b> Top-1 when the family continues</span><span><b>{changeRow ? pct(Number(changeRow.top1_accuracy), 1) : '2.6%'}</b> Top-1 after a real switch</span></div><p className="proof-result selective">At an illustrative validation-median gate, the full model answers {pct(selective.test_coverage)} of leakage-safe events and reaches {pct(selective.top3_accuracy_on_accepted, 1)} Top-3.</p><div className="definition-card"><span>NO PERSONALIZATION CLAIM</span><p>Adding the source-credit label did not improve held-out prediction. Label profiles above are descriptive, not personalized model evidence.</p></div></aside>
      </section>
    </main>
  );
}

export default function HomePage() {
  const [view, setView] = useState<View>('home');
  return <div className="app-shell"><Header view={view} setView={setView} />{view === 'home' && <Home setView={setView} />}{view === 'repertoire' && <RepertoireView />}{view === 'references' && <ReferenceView />}{view === 'rhyme' && <RhymeView />}</div>;
}
