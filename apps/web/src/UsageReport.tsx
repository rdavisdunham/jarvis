export interface CostReport {
  as_of: string;
  tracking_since: string | null;
  last_7_days_usd: number;
  last_30_days_usd: number;
  partial_7_days: boolean;
  partial_30_days: boolean;
  features: { id: string; label: string; last_7_days_usd: number; last_30_days_usd: number; usage_records: number }[];
}
export function costLabel(value: number) {
  if (value > 0 && value < .0001) return "<$0.0001";
  return "$" + value.toFixed(value !== 0 && Math.abs(value) < 0.01 ? 4 : 2);
}
export function UsageReport({ report }: { report: CostReport }) {
  return <div className="usage-report">
    <div className="usage-totals">
      <div><small>Last 7 days</small><strong>{costLabel(report.last_7_days_usd)}</strong>{report.partial_7_days && <small>Partial history</small>}</div>
      <div><small>Last 30 days</small><strong>{costLabel(report.last_30_days_usd)}</strong>{report.partial_30_days && <small>Partial history</small>}</div>
    </div>
    {report.features.length ? <div className="usage-breakdown">
      <table>
        <caption>Recorded AI cost by feature</caption>
        <thead><tr><th scope="col">Feature</th><th scope="col">7 days</th><th scope="col">30 days</th></tr></thead>
        <tbody>{report.features.map(row => <tr key={row.id}><th scope="row">{row.label}</th><td>{costLabel(row.last_7_days_usd)}</td><td>{costLabel(row.last_30_days_usd)}</td></tr>)}</tbody>
      </table>
    </div> : <p className="footnote">No model usage recorded yet. Costs appear as Eri works.</p>}
    <p className="footnote">
      {report.tracking_since ? "Continuous tracking since " + new Date(report.tracking_since).toLocaleDateString() + ". " : ""}
      Estimates from recorded model usage; periods when tracking was off are missing.
      Hosting, provider credits, and isolated eval runs are excluded. Features that do not call an AI model add no AI charge.
    </p>
    <p className="footnote">Check final charges in <a href="https://platform.openai.com/usage" target="_blank" rel="noreferrer">OpenAI Usage</a> or <a href="https://aistudio.google.com/usage" target="_blank" rel="noreferrer">Google AI Studio</a>.</p>
  </div>;
}
