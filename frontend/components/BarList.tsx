type BarItem = {
  label: string;
  value: number;
};

type BarListProps = {
  title: string;
  items: BarItem[];
  format?: (value: number) => string;
};

const BarList = ({ title, items, format }: BarListProps) => {
  const maxValue = Math.max(...items.map((item) => item.value), 1);
  return (
    <div className="card space-y-3">
      <div className="text-sm font-semibold text-slate-700">{title}</div>
      <div className="space-y-2">
        {items.map((item) => (
          <div key={item.label} className="space-y-1">
            <div className="flex items-center justify-between text-xs text-slate-500">
              <span>{item.label}</span>
              <span>{format ? format(item.value) : item.value.toFixed(1)}</span>
            </div>
            <div className="h-2 rounded-full bg-slate-100">
              <div
                className="h-2 rounded-full bg-slate-400"
                style={{ width: `${(item.value / maxValue) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default BarList;
