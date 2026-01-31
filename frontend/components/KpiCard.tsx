type KpiCardProps = {
  label: string;
  value: string;
  helper?: string;
};

const KpiCard = ({ label, value, helper }: KpiCardProps) => {
  return (
    <div className="card">
      <div className="kpi-label">{label}</div>
      <div className="kpi">{value}</div>
      {helper && <div className="mt-1 text-xs text-slate-500">{helper}</div>}
    </div>
  );
};

export default KpiCard;
