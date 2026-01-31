import "./globals.css";
import Navigation from "../components/Navigation";

export const metadata = {
  title: "Drop-off Detective",
  description: "Checkout drop-off analytics and RCA demo",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Navigation />
        <main className="mx-auto max-w-6xl space-y-6 px-6 py-6">{children}</main>
      </body>
    </html>
  );
}
