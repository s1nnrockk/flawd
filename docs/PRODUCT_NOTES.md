# FLAWD product notes

## Future usage-credit system (proposal)

This is a product idea for a later hosted/web version. It is not implemented in the local model or CLI.

- Every new account receives a one-time starting balance of **1,000,000 free credits**.
- The initial proposed thinking rate is **10 credits per second** of billable model thinking.
- Credits are an application billing unit, not tokenizer tokens and not a promise that one credit equals one generated text token.
- The interface should show the estimated charge before a long request and the final charge after it completes.
- The account should have a readable usage ledger: starting grant, each debit, refunds, and remaining balance.
- Failed requests that produce no usable result should not silently consume the user's balance.
- The service must enforce a maximum request budget and stop or ask for confirmation before exceeding it.
- The rate, rounding rule, and what counts as billable thinking time must be disclosed before launch.

### Open implementation decisions

Before enabling real billing, define whether time is rounded per started second or measured fractionally, how queue time is excluded, how interrupted generations are charged, and how the free grant is protected against duplicate-account abuse without invasive data collection.

The initial numbers are provisional product settings and can be changed before launch.
