import { useMemo, useState } from "react";
import { Check, Copy, ExternalLink } from "lucide-react";
import { Button } from "../ui/Button";
import { Segmented } from "../ui/Segmented";
import { Sheet } from "../ui/Sheet";

type Props = {
  open: boolean;
  onClose: () => void;
  defaultCanvas: string;
  defaultPreset: string;
};

function CodeBlock({ label, code }: { label: string; code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-sunken">
      <div className="flex items-center justify-between border-b border-line px-3 py-1.5">
        <span className="text-[10.5px] font-medium uppercase tracking-[0.08em] text-faint">{label}</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={async () => {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1600);
          }}
        >
          {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      <pre className="scrollbar-slim overflow-x-auto p-3 font-mono text-[11.5px] leading-[1.65] text-muted">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function UsageSheet({ open, onClose, defaultCanvas, defaultPreset }: Props) {
  const [tab, setTab] = useState<"api" | "mcp">("api");

  const baseUrl = useMemo(() => {
    const { hostname, origin } = window.location;
    // `npm run dev` proxies the API but serves from localhost, which is not a useful address to copy.
    return hostname === "127.0.0.1" || hostname === "localhost" ? "https://mrfakename-minimax-h3-faster.hf.space" : origin;
  }, []);

  const python = `from gradio_client import Client

client = Client("${baseUrl}")
video, report, refined_prompt = client.predict(
    prompt="A fox running through fresh snow",
    canvas="${defaultCanvas}",
    duration=5,
    generation_preset="${defaultPreset}",
    api_name="/generate",
)
print(video)`;

  const javascript = `import { Client } from "@gradio/client";

const app = await Client.connect("${baseUrl}");
const result = await app.predict("/generate", {
  prompt: "A fox running through fresh snow",
  canvas: "${defaultCanvas}",
  duration: 5,
  generation_preset: "${defaultPreset}",
});
console.log(result.data[0]);`;

  const mcpConfig = `{
  "mcpServers": {
    "minimax-h3": {
      "url": "${baseUrl}/gradio_api/mcp/"
    }
  }
}`;

  return (
    <Sheet
      open={open}
      onClose={onClose}
      title="Use this from code"
      subtitle="The same queued endpoint the studio calls. ZeroGPU usage is attributed to the authenticated caller."
    >
      <div className="flex flex-col gap-4">
        <Segmented
          ariaLabel="Integration method"
          value={tab}
          onChange={setTab}
          options={[
            { value: "api" as const, label: "Gradio API" },
            { value: "mcp" as const, label: "MCP server" },
          ]}
        />

        {tab === "api" ? (
          <>
            <p className="text-[12.5px] leading-[1.6] text-muted">
              The <code className="rounded bg-surface px-1 py-0.5 font-mono text-[11.5px]">/generate</code> endpoint is
              queued and streams progress events, so a client can render the same step counter this page does.
            </p>
            <CodeBlock label="Python" code={python} />
            <CodeBlock label="JavaScript" code={javascript} />
            <Button variant="outline" size="sm" onClick={() => window.open(`${baseUrl}/gradio_api/info`, "_blank")}>
              Full endpoint schema <ExternalLink className="size-3.5" />
            </Button>
          </>
        ) : (
          <>
            <p className="text-[12.5px] leading-[1.6] text-muted">
              Add this remote server to any streamable-HTTP MCP client. The{" "}
              <code className="rounded bg-surface px-1 py-0.5 font-mono text-[11.5px]">generate_video</code> tool takes
              the same arguments and shares the same queue.
            </p>
            <CodeBlock label="MCP configuration" code={mcpConfig} />
            <p className="rounded-xl border border-line bg-surface p-3 text-[12px] leading-[1.6] text-muted">
              Keyframes must be public HTTP(S) URLs when an external agent calls the tool. Text-only generation needs no
              file hosting.
            </p>
            <Button variant="outline" size="sm" onClick={() => window.open(`${baseUrl}/gradio_api/mcp/schema`, "_blank")}>
              MCP tool schema <ExternalLink className="size-3.5" />
            </Button>
          </>
        )}
      </div>
    </Sheet>
  );
}
