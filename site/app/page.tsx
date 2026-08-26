'use client';

import { FormEvent, useMemo, useState } from 'react';
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
  labels: LabelNode[];
  lyricalEdges: LyricalEdge[];
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
    claimBoundary: string;
  };
  ner: {
    entities: Entity[];
    links: ReferenceLink[];
    status: string;
    humanGoldAvailable: boolean;
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

function traitText(trait: Trait): string {
  const high = trait.percentile >= 70;
  const low = trait.percentile <= 30;
  if (trait.key === 'short') return high ? 'mostly shorter written lines' : low ? 'mostly longer written lines' : 'mixed line lengths';
  if (trait.key === 'repeat') return high ? 'frequent exact-line reuse' : low ? 'rare exact-line reuse' : 'moderate line reuse';
  return high ? 'frequent Chinese–English mixing' : low ? 'rare Chinese–English mixing' : 'moderate Chinese–English mixing';
}

function reasonText(edge: LyricalEdge): string {
  const first = edge.reasons[0];
  if (!first) return 'Overall lyrical wording remains close after duplicate controls; no single probe dominates.';
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
    { key: 'repertoire', number: '01', title: 'Who writes nearby?', action: 'Open repertoire map', description: 'Select a source label. The graph returns the closest lyric repertoires and states whether wording, written endings, or writing form explains each link.', proof: 'Inside the fixed corpus, dense + character fusion beats either representation alone.' },
    { key: 'references', number: '02', title: 'Which cultural worlds appear?', action: 'Open reference network', description: 'Move from one source label to a supported lyric reference, then see which other labels invoke the same reference.', proof: 'Visible edges survive shared-text exclusion, uncertainty, and false-discovery-rate control.' },
    { key: 'rhyme', number: '03', title: 'What ending could come next?', action: 'Open rhyme lab', description: 'Enter a Chinese ending to receive ranked next written-ending families, then see which repertoires emphasize a selected family.', proof: 'Sequential context is tested on song-held-out data; label personalization is not claimed.' },
  ];
  return (
    <main className="home-shell">
      <section className="hero"><p className="eyebrow">ONE CORPUS · THREE EVIDENCE-GRADED TASKS</p><h1>What makes a Chinese rap<br /><em>lyrical identity</em> recognizable?</h1><p className="hero-question">{data.question}</p></section>
      <section className="task-grid" aria-label="Result tools">
        {cards.map((card) => <button key={card.key} className={`task-card task-${card.number}`} onClick={() => setView(card.key)}><span className="task-number">{card.number}</span><h2>{card.title}</h2><p>{card.description}</p><strong>{card.proof}</strong><span className="card-action">{card.action} <i>↗</i></span></button>)}
      </section>
      <footer className="home-foot"><span>TEXT-DERIVED · SONG-HELD-OUT · DUPLICATE-CONTROLLED</span><span>No lyric lines are published.</span></footer>
    </main>
  );
}

function RepertoireGraph({ selected, onSelect, selectedEdge, onEdge }: { selected: LabelNode; onSelect: (node: LabelNode) => void; selectedEdge: LyricalEdge | null; onEdge: (edge: LyricalEdge) => void }) {
  const adjacent = useMemo(() => data.lyricalEdges.filter((edge) => edge.a === selected.id || edge.b === selected.id).sort((a, b) => (a.status === b.status ? b.repeatability - a.repeatability : a.status === 'repeatable' ? -1 : 1)).slice(0, 7), [selected.id]);
  const center = { x: 380, y: 278 };
  const points = adjacent.map((edge, index) => {
    const angle = -Math.PI / 2 + index * (Math.PI * 2 / Math.max(adjacent.length, 1));
    const distance = 178 + (1 - edge.repeatability) * 54;
    const other = labelById.get(edge.a === selected.id ? edge.b : edge.a)!;
    return { edge, other, x: center.x + Math.cos(angle) * distance, y: center.y + Math.sin(angle) * distance };
  });
  return (
    <div className="graph-stage">
      <svg viewBox="0 0 760 556" role="img" aria-label={`Closest lyrical repertoires around ${selected.label}`}>
        <defs><filter id="soft"><feDropShadow dx="0" dy="8" stdDeviation="10" floodOpacity="0.24" /></filter></defs>
        {points.map(({ edge, other, x, y }) => { const active = selectedEdge === edge; const color = SIGNAL_COLOR[edge.dominantSignal] ?? SIGNAL_COLOR.semanticOnly; const mx = (x + center.x) / 2; const my = (y + center.y) / 2; return <g key={`${edge.a}-${edge.b}`} className={`graph-edge ${active ? 'active' : ''}`} role="button" tabIndex={0} aria-pressed={active} aria-label={`Explain link to ${other.label}: ${reasonText(edge)}; ${pct(edge.repeatability)} resample repeatability`} onClick={() => onEdge(edge)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onEdge(edge); } }}><circle cx={mx} cy={my} r="22" fill="transparent" /><line x1={center.x} y1={center.y} x2={x} y2={y} stroke={color} strokeWidth={active ? 6 : 2 + edge.repeatability * 3} strokeDasharray={edge.status === 'exploratory' ? '5 8' : undefined} /><circle cx={mx} cy={my} r="13" fill="#11141a" stroke={color} /><text x={mx} y={my + 4} textAnchor="middle" className="edge-percent">{Math.round(edge.repeatability * 100)}</text><title>{`${other.label}: ${reasonText(edge)}; ${pct(edge.repeatability)} resample repeatability`}</title></g>; })}
        {points.map(({ edge, other, x, y }) => <g key={other.id} className="graph-node" transform={`translate(${x} ${y})`} role="button" tabIndex={0} aria-label={`Select ${other.label} repertoire`} onClick={() => onSelect(other)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(other); } }}><circle r={edge.status === 'repeatable' ? 28 : 23} fill={edge.status === 'repeatable' ? '#f4f0e8' : '#20262f'} stroke={SIGNAL_COLOR[edge.dominantSignal] ?? '#8e96a3'} strokeWidth="3" /><text y={edge.status === 'repeatable' ? 44 : 39} textAnchor="middle" className="node-name">{other.label}</text></g>)}
        <g className="center-node" transform={`translate(${center.x} ${center.y})`} filter="url(#soft)"><circle r="72" /><text y="-5" textAnchor="middle">{selected.label}</text><text y="20" textAnchor="middle" className="center-sub">selected repertoire</text></g>
        {adjacent.length === 0 && <text x="380" y="395" textAnchor="middle" className="empty-svg">No stable lyrical neighbour is available for this label.</text>}
      </svg>
      <div className="graph-legend">{Object.entries(SIGNAL_LABEL).map(([key, label]) => <span key={key}><i style={{ background: SIGNAL_COLOR[key] }} />{label}</span>)}<span><b>solid</b> repeatable</span><span><b>dashed</b> exploratory</span></div>
    </div>
  );
}

function RepertoireView() {
  const [selected, setSelected] = useState<LabelNode>(bestConnectedLabel());
  const [selectedEdge, setSelectedEdge] = useState<LyricalEdge | null>(null);
  const adjacent = useMemo(() => data.lyricalEdges.filter((edge) => edge.a === selected.id || edge.b === selected.id).sort((a, b) => (a.status === b.status ? b.repeatability - a.repeatability : a.status === 'repeatable' ? -1 : 1)), [selected.id]);
  const standout = selected.traits.length ? [...selected.traits].sort((a, b) => Math.abs(b.percentile - 50) - Math.abs(a.percentile - 50))[0] : null;
  const fusion = data.retrieval.systems.find((system) => system.name === 'Fusion')!;
  const single = data.retrieval.systems.find((system) => system.name === 'Character TF-IDF')!;
  const openLabel = (label: LabelNode) => { setSelected(label); setSelectedEdge(null); };
  return (
    <main className="tool-shell">
      <section className="tool-intro"><div><p className="eyebrow">RESULT 01 · LANGUAGE</p><h1>Lyrical repertoire map</h1><p>Select one label; distance and links are recomputed from its nearest released matches, so every visible line has an explanation.</p></div><LabelSearch selected={selected} onSelect={openLabel} available={data.labels.filter((label) => label.terms.length > 0)} /></section>
      <section className="workspace two-column"><RepertoireGraph selected={selected} onSelect={openLabel} selectedEdge={selectedEdge} onEdge={setSelectedEdge} /><aside className="result-panel">
        <div className="panel-head"><span className="status-dot verified" /><span>{selected.independentSongs} independent-song profile</span></div><h2>{selected.label}</h2>
        <p className="lead-result">{standout ? `Most distinctive writing habit: ${traitText(standout)}.` : 'This label has reference evidence but not enough material for a full repertoire profile.'} {selected.rhyme ? `Its most common written-ending family is ${selected.rhyme.dominantFamily}.` : ''}</p>
        {selected.terms.length > 0 && <section className="panel-section"><h3>Distinctive wording</h3><div className="term-cloud">{selected.terms.map((term) => <span key={term.text} title={`Supported across ${term.supportSongs} songs`}>{term.text}</span>)}</div></section>}
        {selected.rhyme && <section className="panel-section"><h3>Written-ending fingerprint</h3><div className="family-row">{selected.rhyme.topFamilies.slice(0, 4).map((item) => <div key={item.value}><b>{item.value}</b><span>{pct(item.share)}</span></div>)}</div><p className="meaning-line">Adjacent lines retain the same ending family {pct(selected.rhyme.adjacentSameFamilyRate)} of the time.</p></section>}
        <section className="panel-section"><h3>{selectedEdge ? 'Why this pair is linked' : 'Closest released matches'}</h3>{selectedEdge ? <div className="why-card" style={{ borderColor: SIGNAL_COLOR[selectedEdge.dominantSignal] }}><b>{labelById.get(selectedEdge.a === selected.id ? selectedEdge.b : selectedEdge.a)?.label}</b><p>{reasonText(selectedEdge)}</p><span className={selectedEdge.status === 'repeatable' ? 'evidence strong' : 'evidence provisional'}>{pct(selectedEdge.repeatability)} resample repeatability · {selectedEdge.status}</span><button onClick={() => setSelectedEdge(null)}>Back to all matches</button></div> : adjacent.slice(0, 5).map((edge) => { const other = labelById.get(edge.a === selected.id ? edge.b : edge.a)!; return <button className="match-row" key={`${edge.a}-${edge.b}`} onClick={() => setSelectedEdge(edge)}><span className="signal-bar" style={{ background: SIGNAL_COLOR[edge.dominantSignal] }} /><span><b>{other.label}</b><small>{reasonText(edge)}</small></span><em>{Math.round(edge.repeatability * 100)}</em></button>; })}</section>
      </aside></section>
      <section className="evidence-strip"><div><span>Fixed-corpus holdout</span><b>Fusion puts the correct label in the top 10 for {pct(fusion.recall10.estimate, 1)} of songs.</b></div><div><span>Why fusion</span><b>MRR rises from {single.mrr.estimate.toFixed(3)} to {fusion.mrr.estimate.toFixed(3)} by combining character and BGE-M3 evidence.</b></div><div><span>Boundary</span><b>TF–IDF learns the frozen unlabeled corpus distribution; matches are not friendship, influence, or verified identity.</b></div></section>
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
      <section className="workspace two-column"><ReferenceGraph selected={selected} selectedEntity={selectedEntity} onEntity={setSelectedEntity} /><aside className="result-panel"><div className="panel-head"><span className="status-dot provisional" /><span>Statistically supported · human gold pending</span></div><h2>{selectedEntity?.name ?? selected.label}</h2>{selectedEntity && selectedLink ? <><p className="lead-result">After shared-text exclusion, <b>{selectedEntity.name}</b> appears in {selectedLink.songs} of {selectedLink.labelSongs} eligible <b>{selected.label}</b> song units ({pct(selectedLink.share, 1)}). Its shrunken enrichment is {selectedLink.lift.toFixed(1)}× versus other labels.</p><section className="panel-section"><h3>What the edge means</h3><div className="definition-card"><span>{ENTITY_LABEL[selectedEntity.type] ?? selectedEntity.type}</span><p>A repeated textual reference inside this source-labelled repertoire. It does not establish hometown, affiliation, belief, collaboration, preference, or a real-world relationship.</p></div></section><section className="panel-section"><h3>Who else references it?</h3>{sharedLabels.length ? <div className="shared-list">{sharedLabels.sort((a, b) => b.lift * b.songs - a.lift * a.songs).slice(0, 6).map((link) => <button key={link.labelId} onClick={() => openLabel(labelById.get(link.labelId)!)}><b>{labelById.get(link.labelId)?.label}</b><span>{link.songs}/{link.labelSongs} song units · {link.lift.toFixed(1)}× shrunken enrichment</span></button>)}</div> : <p className="meaning-line">No other source-label edge to this reference survives the same uncertainty and FDR gates.</p>}</section><section className="panel-section"><h3>Reliability</h3><p className="meaning-line">Conservative 95% enrichment interval: {selectedLink.liftLow.toFixed(1)}–{selectedLink.liftHigh.toFixed(1)}×; BH q={qText(selectedLink.qValue)}; class {selectedLink.reliability.toLocaleLowerCase()}. The lexicon and transformer agree on the released surface, but extraction precision/recall remains unknown until dual review.</p></section></> : <p className="lead-result">No entity clears shared-text, support, uncertainty, and FDR gates for this label.</p>}</aside></section>
      <section className="evidence-strip"><div><span>Released layer</span><b>{data.ner.entities.length} provisional surfaces remain after shared-text and semantic gates.</b></div><div><span>Supported relations</span><b>{data.ner.links.length} label–reference edges survive uncertainty and BH-FDR control.</b></div><div><span>Boundary</span><b>Edges encode lyric references—not social relations, biography, or preference.</b></div></section>
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
        <div className="recommendation-card"><div className="recommendation-head"><div><p className="micro-label">RANKED NEXT FAMILIES</p><h2>{family ? `After ${family}` : 'Corpus-wide baseline'}</h2></div><span className="answer-state fallback">{family ? 'observed transition' : 'global baseline'}</span></div><div className="recommendations">{recommendations.slice(0, 5).map((item, index) => { const globalItem = data.rhyme.globalTop5.find((candidate) => candidate.written_rhyme_family === item.written_rhyme_family); const lift = globalItem ? item.probability / globalItem.probability : null; return <button type="button" className={`recommendation ${focusedFamily === item.written_rhyme_family ? 'selected' : ''}`} key={item.written_rhyme_family} onClick={() => setFocusedFamily(item.written_rhyme_family)}><span className="rank">0{index + 1}</span><span className="family"><b>{item.written_rhyme_family}</b><small>Try {FAMILY_EXAMPLES[item.written_rhyme_family]?.join(' / ')}</small></span><span className="probability"><b>{pct(item.probability, 1)}</b><span>{lift === null ? 'outside global top 5' : lift >= 1 ? `${lift.toFixed(1)}× global` : 'below global'}</span></span><span className="prob-bar"><i style={{ width: `${Math.min(100, item.probability * 180)}%` }} /></span></button>; })}</div><p className="recommendation-why">{family ? `Why these results: a smoothed transition model learned from ${markov?.training_event_support.toLocaleString()} training events that originally followed ${family}. Select a result to inspect repertoire evidence.` : 'Enter an ending to use an observed transition.'}</p><section className="family-evidence"><h3>Repertoires emphasizing {evidenceFamily || 'this family'}</h3>{relatedLabels.length ? <div className="family-label-list">{relatedLabels.map((row) => <div key={row.labelId}><b>{labelById.get(row.labelId)?.label}</b><span>{pct(row.share, 1)} of written endings · {Math.pow(2, row.log2RateRatio).toFixed(1)}× corpus rate</span></div>)}</div> : <p>No label clears the public support gate for this family.</p>}<small>Descriptive corpus evidence only—not an intrinsic rapper preference.</small></section></div>
        <aside className="rhyme-proof"><p className="micro-label">HOW RELIABLE?</p><h2>Useful for ranked options.<br />Exact switches remain hard.</h2><div className="metric-stack"><div><span>Global Top-3</span><i><b style={{ width: `${global.top3_accuracy * 100}%` }} /></i><em>{pct(global.top3_accuracy, 1)}</em></div><div><span>One-step Top-3</span><i><b style={{ width: `${markovMetric.top3_accuracy * 100}%` }} /></i><em>{pct(markovMetric.top3_accuracy, 1)}</em></div><div className="highlight"><span>Full-context Top-3</span><i><b style={{ width: `${modelMetric.top3_accuracy * 100}%` }} /></i><em>{pct(modelMetric.top3_accuracy, 1)}</em></div></div><p className="proof-result">Full sequential context adds {pp(modelMetric.top3_accuracy - markovMetric.top3_accuracy)} over the one-step baseline on held-out songs.</p><div className="diagnostic"><span><b>{continueRow ? pct(Number(continueRow.top1_accuracy), 1) : '98.5%'}</b> Top-1 when the family continues</span><span><b>{changeRow ? pct(Number(changeRow.top1_accuracy), 1) : '2.6%'}</b> Top-1 after a real switch</span></div><p className="proof-result selective">At an illustrative validation-median gate, the full model answers {pct(selective.test_coverage)} of leakage-safe events and reaches {pct(selective.top3_accuracy_on_accepted, 1)} Top-3.</p><div className="definition-card"><span>NO PERSONALIZATION CLAIM</span><p>Adding the source-credit label did not improve held-out prediction. Label profiles above are descriptive, not personalized model evidence.</p></div></aside>
      </section>
    </main>
  );
}

export default function HomePage() {
  const [view, setView] = useState<View>('home');
  return <div className="app-shell"><Header view={view} setView={setView} />{view === 'home' && <Home setView={setView} />}{view === 'repertoire' && <RepertoireView />}{view === 'references' && <ReferenceView />}{view === 'rhyme' && <RhymeView />}</div>;
}
