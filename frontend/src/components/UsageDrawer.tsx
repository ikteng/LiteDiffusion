import { Button, Chip, Drawer, Tabs } from "@heroui/react";
import { Check, Clipboard, Code2, ExternalLink, PlugZap } from "lucide-react";
import { useMemo, useState } from "react";

type Props = {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
};

function CodeBlock({ code, label }: { code: string; label: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <div className="overflow-hidden rounded-xl border border-divider bg-background">
      <div className="flex items-center justify-between border-b border-divider px-3 py-2">
        <span className="text-[11px] font-medium uppercase tracking-wider text-muted">{label}</span>
        <Button size="sm" variant="ghost" onPress={copy}>
          {copied ? <Check className="size-3.5" /> : <Clipboard className="size-3.5" />}
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      <pre className="overflow-x-auto p-4 text-xs leading-6 text-foreground"><code>{code}</code></pre>
    </div>
  );
}

export function UsageDrawer({ isOpen, onOpenChange }: Props) {
  const baseUrl = useMemo(() => {
    if (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost") {
      return "https://mrfakename-minimax-h3-faster.hf.space";
    }
    return window.location.origin;
  }, []);

  const python = `from gradio_client import Client

client = Client("${baseUrl}")
video, report, refined_prompt = client.predict(
    prompt="A fox running through fresh snow",
    canvas="960x544 · 16:9 fast",
    duration=5,
    generation_preset="Balanced — best overall (recommended)",
    api_name="/generate",
)
print(video)`;

  const javascript = `import { Client } from "@gradio/client";

const app = await Client.connect("${baseUrl}");
const result = await app.predict("/generate", {
  prompt: "A fox running through fresh snow",
  canvas: "960x544 · 16:9 fast",
  duration: 5,
  generation_preset: "Balanced — best overall (recommended)",
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
    <Drawer isOpen={isOpen} onOpenChange={onOpenChange}>
      <Drawer.Backdrop>
        <Drawer.Content placement="right" className="sm:max-w-xl">
          <Drawer.Dialog>
            <Drawer.CloseTrigger />
            <Drawer.Header>
              <Drawer.Heading>Use MiniMax-H3 elsewhere</Drawer.Heading>
            </Drawer.Header>
            <Drawer.Body className="space-y-5">
              <p className="text-sm leading-6 text-muted">
                Use the same queued generation endpoint from code or connect it directly to an MCP-capable agent.
                ZeroGPU usage is attributed through Hugging Face when the client is authenticated.
              </p>

              <Tabs aria-label="Usage method" defaultSelectedKey="api" variant="secondary">
                <Tabs.ListContainer>
                  <Tabs.List>
                    <Tabs.Tab id="api"><Code2 className="size-4" /> API</Tabs.Tab>
                    <Tabs.Tab id="mcp"><PlugZap className="size-4" /> MCP</Tabs.Tab>
                  </Tabs.List>
                </Tabs.ListContainer>
                <Tabs.Panel id="api" className="space-y-4 pt-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-semibold">Gradio API</h3>
                      <p className="mt-1 text-xs text-muted">Endpoint <code>/generate</code> · queued · progress events enabled</p>
                    </div>
                    <Chip size="sm" variant="soft" color="success">Live</Chip>
                  </div>
                  <CodeBlock label="Python" code={python} />
                  <CodeBlock label="JavaScript" code={javascript} />
                  <Button variant="secondary" size="sm" onPress={() => window.open(`${baseUrl}/gradio_api/info`, "_blank")}>
                    Full endpoint schema <ExternalLink className="size-3.5" />
                  </Button>
                </Tabs.Panel>
                <Tabs.Panel id="mcp" className="space-y-4 pt-4">
                  <div>
                    <h3 className="text-sm font-semibold">Model Context Protocol</h3>
                    <p className="mt-1 text-xs leading-5 text-muted">
                      Add this remote server to Cursor, Claude, Cline, or another streamable-HTTP MCP client. The
                      <code className="mx-1">generate_video</code> tool exposes the same defaults and queue.
                    </p>
                  </div>
                  <CodeBlock label="MCP configuration" code={mcpConfig} />
                  <div className="rounded-xl border border-divider bg-default/5 p-4 text-xs leading-5 text-muted">
                    Image inputs must be public HTTP(S) URLs when an external agent calls the remote tool. Text-only
                    generation needs no file hosting.
                  </div>
                  <Button variant="secondary" size="sm" onPress={() => window.open(`${baseUrl}/gradio_api/mcp/schema`, "_blank")}>
                    View MCP tool schema <ExternalLink className="size-3.5" />
                  </Button>
                </Tabs.Panel>
              </Tabs>
            </Drawer.Body>
            <Drawer.Footer>
              <Button variant="ghost" onPress={() => onOpenChange(false)}>Close</Button>
            </Drawer.Footer>
          </Drawer.Dialog>
        </Drawer.Content>
      </Drawer.Backdrop>
    </Drawer>
  );
}
