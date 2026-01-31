import BarList from "../components/BarList";
import KpiCard from "../components/KpiCard";
import TrendChart from "../components/TrendChart";
import { fetchJson } from "../lib/api";

export const dynamic = "force-dynamic";

type OverviewMetric = {
  date: string;
  conversion_rate: number;
  abandonment_rate: number;
  payment_fail_rate: number;
  prepaid_share: number;
  aov: number;
};

type RootCauseResponse = {
  model_features: { feature: string; importance: number }[];
};

export default async function DashboardPage() {
  const overview = await fetchJson<OverviewMetric[]>("/metrics/overview");
  const latest = overview[overview.length - 1];
  const response = await fetchJson<RootCauseResponse>("/root_causes");
  const featureItems = response.model_features.map((item) => ({
    label: item.feature,
    value: Math.abs(item.importance),
  }));

  const trendData = overview.slice(-7).map((item) => ({
    label: item.date.slice(5),
    value: item.conversion_rate * 100,
  }));
  const abandonmentTrend = overview.slice(-7).map((item) => ({
    label: item.date.slice(5),
    value: item.abandonment_rate * 100,
  }));

  return (
    <div className="space-y-6">
      <section className="grid gap-4 md:grid-cols-5">
        <KpiCard label="Conversion" value={`${(latest.conversion_rate * 100).toFixed(1)}%`} />
        <KpiCard label="Abandonment" value={`${(latest.abandonment_rate * 100).toFixed(1)}%`} />
        <KpiCard label="Payment Fail" value={`${(latest.payment_fail_rate * 100).toFixed(1)}%`} />
        <KpiCard label="Prepaid Share" value={`${(latest.prepaid_share * 100).toFixed(1)}%`} />
        <KpiCard label="AOV" value={`₹${latest.aov.toFixed(0)}`} helper="Last 24h" />
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <TrendChart
          title="Conversion trend (%)"
          data={trendData}
          format={(value) => `${value.toFixed(1)}%`}
        />
        <TrendChart
          title="Abandonment trend (%)"
          data={abandonmentTrend}
          format={(value) => `${value.toFixed(1)}%`}
        />
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        <div className="card space-y-2">
          <div className="text-sm font-semibold text-slate-700">Insights</div>
          <p className="text-sm text-slate-500">
            The model flags the top drivers of abandonment so you can prioritize fixes.
          </p>
          <p className="text-xs text-slate-400">
            Drill into Root Causes for narrative evidence and experiment plans.
          </p>
        </div>
        <div className="lg:col-span-2">
          <BarList
            title="Top predictive features"
            items={featureItems.slice(0, 6)}
            format={(value) => value.toFixed(2)}
          />
        </div>
      </section>
    </div>
  );
}
