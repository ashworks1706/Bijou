import Image from "next/image";

export default function Home() {
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="border-b border-gray-800">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Image
              src="/bijou-logo.svg"
              alt="Bijou"
              width={100}
              height={36}
            />
          </div>
          <a
            href="https://github.com/ashworks1706/Bijou"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm hover:underline"
          >
            GitHub
          </a>
        </div>
      </header>

      {/* Hero */}
      <section className="max-w-4xl mx-auto px-6 py-24 text-center">
        <h1 className="text-5xl sm:text-6xl font-bold mb-6">
          Tiny language models
          <br />
          for tiny devices
        </h1>
        <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
          On-device AI optimized for headphones, wearables, AR glasses, and low-power hardware. 
          Zero cloud. Zero latency. Zero hallucination.
        </p>
        <div className="flex gap-4 justify-center">
          <a
            href="https://github.com/ashworks1706/Bijou"
            className="px-6 py-3 bg-white text-black rounded-lg font-medium hover:opacity-90 transition"
          >
            Get Started
          </a>
          <a
            href="#features"
            className="px-6 py-3 border border-gray-700 rounded-lg font-medium hover:bg-gray-900 transition"
          >
            Learn More
          </a>
        </div>
      </section>

      {/* Example */}
      <section className="max-w-4xl mx-auto px-6 py-16">
        <div className="bg-gray-900 rounded-lg p-8 border border-gray-800">
          <div className="mb-4">
            <span className="text-sm text-gray-400">User command:</span>
            <p className="text-lg mt-1">&quot;turn noise cancelling to high&quot;</p>
          </div>
          <div>
            <span className="text-sm text-gray-400">Bijou output:</span>
            <pre className="mt-2 text-sm bg-black p-4 rounded border border-gray-800 overflow-x-auto">
{`{
  "function": "set_anc_mode",
  "mode": "high"
}`}
            </pre>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="max-w-6xl mx-auto px-6 py-24">
        <h2 className="text-3xl font-bold text-center mb-16">What&apos;s included</h2>
        <div className="grid md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-xl font-semibold mb-3">Synthetic Dataset Generation</h3>
            <p className="text-gray-400">
              Generate command → function-call pairs using teacher models for device-specific training.
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-3">Schema-Constrained Decoding</h3>
            <p className="text-gray-400">
              Ensures every output is valid, typed, and deterministic. No hallucinations.
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-3">Quantized Inference</h3>
            <p className="text-gray-400">
              Export to int8/int4 for fast local inference on resource-constrained devices.
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-3">Tool-Schema Definitions</h3>
            <p className="text-gray-400">
              Define the full list of actions your target device supports with JSON schemas.
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-3">Small-Model Fine-Tuning</h3>
            <p className="text-gray-400">
              Train 1-4B parameter models to output structured JSON only for your use case.
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-3">Browser Demo</h3>
            <p className="text-gray-400">
              Test the full flow: mic → STT → model → JSON → simulated device actions.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 mt-24">
        <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-sm text-gray-400">
            MIT License — Bijou 2025
          </p>
          <a
            href="https://github.com/ashworks1706/Bijou"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm hover:underline"
          >
            View on GitHub →
          </a>
        </div>
      </footer>
    </div>
  );
}
