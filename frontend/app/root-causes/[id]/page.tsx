import ExportButton from "../../../components/ExportButton";
import { fetchJson } from "../../../lib/api";

export const dynamic = "force-dynamic";

type RootCauseDetail = {
  cause: {
    id: string;
    dimension: string;
    segment: string;
    abandonment_rate: number;
    baseline_rate: number;
    lift: number;
    z_score: number;
    sample_size: number;
    narrative: {
      title: string;
      evidence: string[];
      estimated_impact: string;
      recommended_fix: string;
      experiment_plan: Record<string, unknown>;
    };
  };
};

export default async function RootCauseDetailPage({ params }: { params: { id: string } }) {
  const response = await fetchJson<RootCauseDetail>(`/root_causes/${params.id}`);
  const cause = response.cause;

  return (
    <div className="space-y-6">
      <section className="card space-y-3">
        <div className="text-sm font-semibold text-slate-800">{cause.narrative.title}</div>
        <div className="text-xs text-slate-500">
          {cause.dimension}: {cause.segment} · Lift {(cause.lift * 100).toFixed(1)}% · Z {cause.z_score.toFixed(2)} ·
          Sessions {cause.sample_size}
        </div>
        <ul className="space-y-1 text-sm text-slate-600">
          {cause.narrative.evidence.map((item) => (
            <li key={item}>• {item}</li>
          ))}
        </ul>
        <p className="text-sm text-slate-600">{cause.narrative.estimated_impact}</p>
        <p className="text-xs text-slate-500">{cause.narrative.recommended_fix}</p>
        <ExportButton filename={`experiment-${cause.id}.json`} data={cause.narrative.experiment_plan} />
      </section>

      <section className="card">
        <div className="mb-3 text-sm font-semibold text-slate-700">Experiment JSON</div>
        <pre className="overflow-x-auto rounded bg-slate-900 p-4 text-xs text-slate-100">
          {JSON.stringify(cause.narrative.experiment_plan, null, 2)}
        </pre>
      </section>
    </div>
  );
}
