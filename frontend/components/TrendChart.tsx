type TrendChartProps = {
  title: string;
  data: { label: string; value: number }[];
  format?: (value: number) => string;
};

const TrendChart = ({ title, data, format }: TrendChartProps) => {
  const maxValue = Math.max(...data.map((item) => item.value), 1);
  return (
    <div className="card">
      <div className="mb-3 text-sm font-semibold text-slate-700">{title}</div>
      <div className="flex items-end gap-2">
        {data.map((item) => (
          <div key={item.label} className="flex flex-1 flex-col items-center">
            <div
              className="w-full rounded bg-slate-300"
              style={{ height: `${(item.value / maxValue) * 120 + 8}px` }}
            />
            <div className="mt-2 text-[10px] text-slate-400">{item.label}</div>
            <div className="text-[10px] text-slate-500">
              {format ? format(item.value) : item.value.toFixed(1)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TrendChart;
