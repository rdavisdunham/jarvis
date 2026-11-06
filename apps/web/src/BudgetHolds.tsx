/** One quiet line for model work still in progress. Nothing shows when there is none worth a cent. */
export function BudgetHolds({ usd }: { usd: number }) {
  if (!(usd >= 0.005)) return null;
  return (
    <p className="footnote">
      Plus ${usd.toFixed(2)} set aside for work that is still finishing.
    </p>
  );
}

/** This calendar month: recorded spend, the month-end projection, and work still in progress. */
export function MonthSpend({
  spent,
  limit,
  projected,
  inFlight,
}: {
  spent: number;
  limit?: number;
  projected: number | null;
  inFlight: number;
}) {
  return (
    <>
      <div className="budget-line">
        <strong>
          ${spent.toFixed(2)}
          <span>{limit != null && " / $" + limit.toFixed(0)} spent this month</span>
        </strong>
        {projected != null && (
          <span>At this month's pace: about ${projected.toFixed(2)} by month end</span>
        )}
      </div>
      <BudgetHolds usd={inFlight} />
    </>
  );
}
