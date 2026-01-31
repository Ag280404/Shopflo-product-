import Link from "next/link";
import { fetchJson } from "../../lib/api";

export const dynamic = "force-dynamic";

type RootCauseItem = {
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
  };
};

type RootCauseResponse = {
  causes: RootCauseItem[];
  anomalies: string[];
};

export default async function RootCausesPage({
  searchParams,
}: {
  searchParams: { start_date?: string; end_date?: string; device?: string; payment_method?: string };
}) {
  const params = new URLSearchParams();
  if (searchParams.start_date) params.set("start_date", searchParams.start_date);
  if (searchParams.end_date) params.set("end_date", searchParams.end_date);
  if (searchParams.device) params.set("device", searchParams.device);
  if (searchParams.payment_method) params.set("payment_method", searchParams.payment_method);
  const response = await fetchJson<RootCauseResponse>(`/root_causes?${params.toString()}`);

  return (
    <div className="space-y-6">
      <section className="card">
        <form className="grid gap-4 md:grid-cols-4" action="/root-causes" method="get">
          <div>
            <label className="text-xs text-slate-500">Start date</label>
            <input
              type="date"
              name="start_date"
              defaultValue={searchParams.start_date}
              className="mt-1 w-full rounded border border-slate-200 px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-slate-500">End date</label>
            <input
              type="date"
              name="end_date"
              defaultValue={searchParams.end_date}
              className="mt-1 w-full rounded border border-slate-200 px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="text-xs text-slate-500">Device</label>
            <select
              name="device"
              defaultValue={searchParams.device}
              className="mt-1 w-full rounded border border-slate-200 px-3 py-2 text-sm"
            >
              <option value="">All</option>
              <option value="mobile">Mobile</option>
              <option value="desktop">Desktop</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-500">Payment method</label>
            <select
              name="payment_method"
              defaultValue={searchParams.payment_method}
              className="mt-1 w-full rounded border border-slate-200 px-3 py-2 text-sm"
            >
              <option value="">All</option>
              <option value="COD">COD</option>
              <option value="UPI_INTENT">UPI Intent</option>
              <option value="UPI_COLLECT">UPI Collect</option>
              <option value="CARD">Card</option>
            </select>
          </div>
          <button
            type="submit"
            className="col-span-full rounded bg-slate-900 px-4 py-2 text-sm text-white"
          >
            Apply filters
          </button>
        </form>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        {response.causes.map((cause) => (
          <div key={cause.id} className="card space-y-3">
            <div className="text-sm font-semibold text-slate-800">{cause.narrative.title}</div>
            <div className="text-xs text-slate-500">
              {cause.dimension}: {cause.segment} · Lift {(cause.lift * 100).toFixed(1)}% · Z {cause.z_score.toFixed(2)}
            </div>
            <ul className="space-y-1 text-sm text-slate-600">
              {cause.narrative.evidence.map((item) => (
                <li key={item}>• {item}</li>
              ))}
            </ul>
            <p className="text-xs text-slate-500">{cause.narrative.recommended_fix}</p>
            <Link
              href={`/root-causes/${cause.id}`}
              className="text-xs font-semibold text-slate-700"
            >
              View RCA detail →
            </Link>
          </div>
        ))}
      </section>

      <section className="card space-y-2">
        <div className="text-sm font-semibold text-slate-700">Anomaly detection</div>
        <ul className="text-sm text-slate-600">
          {response.anomalies.map((item) => (
            <li key={item}>• {item}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
