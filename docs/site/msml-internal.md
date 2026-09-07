# msml-team data locations (internal)

The shipped `runcmp/lineup.json` uses `<YOUR-SITE-DATA>` placeholders so the
package carries no personal or site directories. If you can read the msml
shared directories (most of the maintainer's team can), the real locations:

- `<YOUR-SITE-DATA>/etf_rfq_research_dataset`
  is really
  `/v/global/user/y/yu/yuriyn/PycharmProjects5/mstech-alphalab/etf_rfq_research_dataset`
- `<YOUR-SITE-DATA>/payup/dataFullNoDups.csv`
  is really
  `/v/region/na/appl/spg/shared/data/spgrisk/agency/x42/specpricer/dailyDump/v9/prod/dataFullNoDups.csv`

Point the packaged lineup at them in one command:

```bash
sed -i 's|<YOUR-SITE-DATA>/etf_rfq_research_dataset|/v/global/user/y/yu/yuriyn/PycharmProjects5/mstech-alphalab/etf_rfq_research_dataset|g; s|<YOUR-SITE-DATA>/payup/dataFullNoDups.csv|/v/region/na/appl/spg/shared/data/spgrisk/agency/x42/specpricer/dailyDump/v9/prod/dataFullNoDups.csv|g' runcmp/lineup.json
```

The finished showcase reports (with charts) are on the same share —
the top README's "end product" section carries the Explorer paths.
