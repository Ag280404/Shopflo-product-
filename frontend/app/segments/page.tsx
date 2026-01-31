import BarList from "../../components/BarList";
import { fetchJson } from "../../lib/api";

export const dynamic = "force-dynamic";

type SegmentMetric = {
  segment: string;
  value: string;
  total_sessions: number;
  abandonment_rate: number;
  conversion_rate: number;
};

export default async function SegmentExplorerPage() {
  const segments = await fetchJson<SegmentMetric[]>("/metrics/segments");
  const deviceSegments = segments
    .filter((item) => item.segment === "device")
    .map((item) => ({ label: item.value, value: item.abandonment_rate * 100 }));
  const paymentSegments = segments
    .filter((item) => item.segment === "payment_method")
    .map((item) => ({ label: item.value, value: item.abandonment_rate * 100 }));

  return (
    <div className="space-y-6">
      <section className="grid gap-4 lg:grid-cols-2">
        <BarList title="Abandonment by device" items={deviceSegments} format={(value) => `${value.toFixed(1)}%`} />
        <BarList
          title="Abandonment by payment method"
          items={paymentSegments}
          format={(value) => `${value.toFixed(1)}%`}
        />
      </section>

      <section className="card">
        <div className="mb-4 text-sm font-semibold text-slate-700">Segment explorer table</div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-slate-200 text-xs uppercase text-slate-400">
              <tr>
                <th className="py-2 pr-4">Segment</th>
                <th className="py-2 pr-4">Value</th>
                <th className="py-2 pr-4">Sessions</th>
                <th className="py-2 pr-4">Conversion</th>
                <th className="py-2">Abandonment</th>
              </tr>
            </thead>
            <tbody>
              {segments.map((item) => (
                <tr key={`${item.segment}-${item.value}`} className="border-b border-slate-100">
                  <td className="py-2 pr-4 text-slate-600">{item.segment}</td>
                  <td className="py-2 pr-4 text-slate-700">{item.value}</td>
                  <td className="py-2 pr-4 text-slate-500">{item.total_sessions}</td>
                  <td className="py-2 pr-4 text-slate-500">{(item.conversion_rate * 100).toFixed(1)}%</td>
                  <td className="py-2 text-slate-500">{(item.abandonment_rate * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
