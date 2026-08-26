import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Verseprint — Chinese Rap Evidence Lab',
  description:
    'Explore lyrical-repertoire matches, cultural-reference networks, and written-rhyme recommendations derived from a Chinese rap lyrics corpus.',
  openGraph: {
    title: 'Verseprint — Chinese Rap Evidence Lab',
    description: 'Three evidence-bounded tools for lyrical repertoire, cultural reference, and written rhyme.',
    images: [{
      url: '/images/verseprint-social-preview.png',
      width: 1672,
      height: 941,
      alt: 'Abstract fingerprint, network nodes, and colored transition paths on an off-white field.',
    }],
  },
  twitter: {
    card: 'summary_large_image',
    images: ['/images/verseprint-social-preview.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
