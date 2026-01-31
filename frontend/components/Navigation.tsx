import Link from "next/link";

const Navigation = () => {
  return (
    <nav className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4">
      <div className="text-lg font-semibold">Drop-off Detective</div>
      <div className="flex gap-4 text-sm text-slate-600">
        <Link className="hover:text-slate-900" href="/">
          Dashboard
        </Link>
        <Link className="hover:text-slate-900" href="/root-causes">
          Root Causes
        </Link>
        <Link className="hover:text-slate-900" href="/segments">
          Segment Explorer
        </Link>
      </div>
    </nav>
  );
};

export default Navigation;
