# Forecast provenance

**Maintainer’s statement.** I completed and issued each of the 104 match forecasts before its own kickoff.

The tables below document which forecast values a reader can trace to a dated file or GitHub push. A batch export has its own timestamp. For fixtures without a separately preserved first-issue time, the maintainer’s statement is the source for when the forecast was issued. All times are UTC.

For **87 of 104 fixtures**, a pre-kickoff stage generation record or an earlier matching snapshot covers the archived win/draw/loss (W/D/L) percentages, five ordered scorelines and their displayed probabilities. **At least 42 of those 87** can also be checked against a GitHub push before kickoff, including 34 group games and all eight round-of-16 games. The retained stage export predates kickoff for 68 fixtures. Each count measures the records available for verification.

The 17 fixtures outside that coverage are group round 1 matches 1–4, group round 2 matches 26–28, 30–31 and 33–36, and all four quarterfinals. Some have earlier W/D/L values. The fixture ledger identifies those partial matches.

## Original files and generation times

Each stage links to the exact CSV used by the README. The timestamp is the original companion JSON’s `generated_at`. “Before kickoff” compares that one batch timestamp with each fixture’s kickoff. Some forecasts also survive in an earlier file.

| Stage and archived CSV | Retained export generated (UTC) | First kickoff (UTC) | Fixtures with export before kickoff |
|---|---|---|---:|
| [Group round 1](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c8a98448494568c787e7733b4d18ca5aef5030ef/output/predictions/live/group_stage_round1_predictions__v24_abcd_no_e.csv) | 2026-06-14 05:28:15.444856 | 2026-06-11 19:00 | 16/24 |
| [Group round 2](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c8a98448494568c787e7733b4d18ca5aef5030ef/output/predictions/live/group_stage_round2_predictions_clean.csv) | 2026-06-24 04:11:05.609376 | 2026-06-18 16:00 | 0/24 |
| [Group round 3](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c8a98448494568c787e7733b4d18ca5aef5030ef/output/predictions/live/group_stage_round3_predictions_clean.csv) | 2026-06-24 04:14:47.412795 | 2026-06-24 19:00 | 24/24 |
| [Round of 32](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c8a98448494568c787e7733b4d18ca5aef5030ef/output/predictions/round_of_32_predictions_after_group_stage_clean_with_et_penalties.csv) | 2026-06-28 08:42:56 | 2026-06-28 19:00 | 16/16 |
| [Round of 16](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c8a98448494568c787e7733b4d18ca5aef5030ef/output/predictions/round_of_16_predictions_after_r32_clean.csv) | 2026-07-04 05:07:30 | 2026-07-04 17:00 | 8/8 |
| [Quarterfinals](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c8a98448494568c787e7733b4d18ca5aef5030ef/output/predictions/R8_predictions_clean.csv) | 2026-07-12 12:51:42 | 2026-07-09 20:00 | 0/4 |
| [Semifinals](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c8a98448494568c787e7733b4d18ca5aef5030ef/output/predictions/semifinal_predictions_clean.csv) | 2026-07-14 04:03:19 | 2026-07-14 19:00 | 2/2 |
| [Third place and final](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c8a98448494568c787e7733b4d18ca5aef5030ef/output/predictions/third_place_and_final_predictions_clean.csv) | 2026-07-18 04:46:47 | 2026-07-18 21:00 | 2/2 |

The separate [48-team tournament simulation](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c8a98448494568c787e7733b4d18ca5aef5030ef/output/simulations/tournament_simulation_results_default.csv) contains 100,000 runs. Its original ZIP entry records a UTC modification time of **2026-06-11 18:20:06**, 39 minutes 54 seconds before the opener. The archive supplies no separate generation or publication timestamp for this file. It contributes no rows to the 104-match count.

## Dated copies and the fields they support

For each push below, GitHub’s repository activity identifies the server timestamp, event ID and commit. Opening the file at that commit recovers the values available in the repository by that time. The record does not identify who received a forecast. [GitHub activity, oldest first](https://api.github.com/repos/KurokawaMasaya/fifa_world_cup_2026/activity?direction=asc&per_page=100)

| Key | GitHub push (UTC) | Activity ID | File at that commit |
|---|---|---:|---|
| A | 2026-06-11 19:44:30 | `35476295625` | [June 11 group baseline](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/6b29e0b4330dc7ee7f4dae7d10f74de7fa692677/output/predictions/group_stage_predictions_clean.csv) |
| B | 2026-06-14 02:43:52 | `35578160582` | [V2.4 static group forecasts](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/cd339c99dd80024c2c39fe12468fd3f8dbd5d9e1/output/predictions/group_stage_predictions_clean.csv) |
| C | 2026-06-17 09:39:01 | `35745383975` | [Round 1 CSV](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/9400cfd610040ef3706ba636fb3cc6fd015f590f/output/predictions/live/group_stage_round1_predictions__v24_abcd_no_e.csv) |
| D | 2026-06-21 08:32:36 | `35936692841` | [Round 2 CSV](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/333d3e58d6d6a0601e75f70658c882dddf2eed65/output/predictions/live/group_stage_round2_predictions_clean.csv) |
| E | 2026-07-04 05:37:27 | `36612576388` | [Round of 16 CSV](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c21c729f54eb5130509dffed8cbf0ec5c2a3254e/output/predictions/round_of_16_predictions_after_r32_clean.csv) |

**Group round 1.** An earlier V2 table (key **L**) contains all 24 archived W/D/L triples when rounded to whole percentages. Its original ZIP modification time is 2026-06-11 16:45:18 UTC, before Mexico–South Africa at 19:00. The June 11 push containing those same bytes arrived at 19:44:30. The V2 table has one score candidate. The later 72-match production table (key **P**) matches all 24 archived round-1 W/D/L triples, score lists and displayed percentages. Its metadata records 2026-06-13 12:17:24.649856 UTC, before matches 5–24. The first four matches had already started. [V2 table](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/6b29e0b4330dc7ee7f4dae7d10f74de7fa692677/output/predictions/group_stage_predictions_v2_latest.csv) · [Production table](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/cd339c99dd80024c2c39fe12468fd3f8dbd5d9e1/output/predictions/group_stage_predictions_clean.csv) · [Production metadata](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/cd339c99dd80024c2c39fe12468fd3f8dbd5d9e1/output/predictions/group_stage_predictions_clean__metadata.json)

**Group rounds 2 and 3.** The June 21 round-2 export records generation at 2026-06-21 08:30:38.916647 UTC. Its GitHub copy matches every shared column in all 24 retained rows. Matches 37–48 were still to come at the time of the push. For three earlier matches (25, 29 and 32), the June 14 snapshot already has matching W/D/L and all five displayed scores and percentages in a static forecast. The static and rolling versions used different rating states. The June 24 round-3 metadata predates all 24 fixtures. The README documents changed score candidates in rounds 2 and 3. In 20 slots in round 2 and 14 in round 3, the displayed percentage still refers to the previous candidate. A field match here means that the displayed values agree. [June 21 metadata](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/333d3e58d6d6a0601e75f70658c882dddf2eed65/output/predictions/live/group_stage_round2_predictions__metadata.json) · [README analysis](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c8a98448494568c787e7733b4d18ca5aef5030ef/README.md#historical-evaluation)

**Knockouts.** The July 4 push contains the exact round-of-32 and round-of-16 CSVs. It followed the last round-of-32 kickoff and preceded the first round-of-16 kickoff by 11 hours 22 minutes 33 seconds. The semifinals and both medal matches have pre-kickoff generation records in the original archive. The reviewed GitHub history first shows these three later-stage tables on November 19. [July 4 round-of-32 file](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c21c729f54eb5130509dffed8cbf0ec5c2a3254e/output/predictions/round_of_32_predictions_after_group_stage_clean_with_et_penalties.csv) · [November archive commit](https://github.com/KurokawaMasaya/fifa_world_cup_2026/commit/11992ff4778fd26abb6a14d4236d58294095c192)

**Quarterfinals.** The maintainer confirms all four forecasts were completed before their respective kickoffs. The original export script writes `R8_predictions_clean.csv` and then records the current UTC time in `R8_predictions_clean__metadata.json`. Each run writes to the same paths. The retained JSON dates that export to July 12 at 12:51:42 UTC. The last quarterfinal had kicked off at 01:00 UTC that day. The exact times of first issue are not recorded in the surviving files.

## Match-by-match record

“Retained CSV export” shows whether the archived batch was written before or after kickoff. The maintainer’s statement above applies to all rows. The GitHub column gives the most detailed matching record found before that fixture. **CSV** means identical file bytes, **W/D/L + five** compares the three regulation-time percentages and the five ordered scores with their five displayed percentages, and **W/D/L** covers the three outcome percentages alone. The last column identifies additional evidence from the earlier local group files **L** and **P**. The local sources use their recorded export or file modification times. A dash means no qualifying record in that column.

Match links use the ESPN event ID from the original `output/live/fixtures_results.csv` and kickoff uses its `kickoff_time_utc` field. Group rows were joined by the ordered team pairing after resolving name variants. Knockout rows use the event ID directly. For Mexico–England (760505), the forecast and July 4 schedule show 00:00 on July 6 and the final fixture log shows 01:00. The ledger uses 01:00. Both are after the July 4 push. [July 4 fixture log](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c21c729f54eb5130509dffed8cbf0ec5c2a3254e/output/live/fixtures_results.csv)

<details>
<summary>Group stage (72 matches)</summary>

| Stage | CSV match_id | Fixture (ESPN) | Kickoff (UTC, 2026) | Retained CSV export | GitHub before kickoff | Earlier local group file |
|---|---:|---|---|---|---|---|
| G1 | 1 | [Mexico – South Africa](https://www.espn.com/soccer/match/_/gameId/760415) | 06-11 19:00 | After | — | L · W/D/L |
| G1 | 2 | [South Korea – Czechia](https://www.espn.com/soccer/match/_/gameId/760414) | 06-12 02:00 | After | A · W/D/L | L · W/D/L |
| G1 | 3 | [Canada – Bosnia-Herzegovina](https://www.espn.com/soccer/match/_/gameId/760416) | 06-12 19:00 | After | A · W/D/L | L · W/D/L |
| G1 | 4 | [United States – Paraguay](https://www.espn.com/soccer/match/_/gameId/760417) | 06-13 01:00 | After | A · W/D/L | L · W/D/L |
| G1 | 5 | [Qatar – Switzerland](https://www.espn.com/soccer/match/_/gameId/760420) | 06-13 19:00 | After | A · W/D/L | P · W/D/L + five |
| G1 | 6 | [Brazil – Morocco](https://www.espn.com/soccer/match/_/gameId/760419) | 06-13 22:00 | After | A · W/D/L | P · W/D/L + five |
| G1 | 7 | [Haiti – Scotland](https://www.espn.com/soccer/match/_/gameId/760418) | 06-14 01:00 | After | A · W/D/L | P · W/D/L + five |
| G1 | 8 | [Australia – Türkiye](https://www.espn.com/soccer/match/_/gameId/760421) | 06-14 04:00 | After | B · W/D/L + five | P · W/D/L + five |
| G1 | 9 | [Germany – Curaçao](https://www.espn.com/soccer/match/_/gameId/760422) | 06-14 17:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 10 | [Netherlands – Japan](https://www.espn.com/soccer/match/_/gameId/760425) | 06-14 20:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 11 | [Ivory Coast – Ecuador](https://www.espn.com/soccer/match/_/gameId/760423) | 06-14 23:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 12 | [Sweden – Tunisia](https://www.espn.com/soccer/match/_/gameId/760424) | 06-15 02:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 13 | [Spain – Cape Verde](https://www.espn.com/soccer/match/_/gameId/760428) | 06-15 16:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 14 | [Belgium – Egypt](https://www.espn.com/soccer/match/_/gameId/760426) | 06-15 19:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 15 | [Saudi Arabia – Uruguay](https://www.espn.com/soccer/match/_/gameId/760429) | 06-15 22:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 16 | [Iran – New Zealand](https://www.espn.com/soccer/match/_/gameId/760427) | 06-16 01:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 17 | [France – Senegal](https://www.espn.com/soccer/match/_/gameId/760432) | 06-16 19:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 18 | [Iraq – Norway](https://www.espn.com/soccer/match/_/gameId/760430) | 06-16 22:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 19 | [Argentina – Algeria](https://www.espn.com/soccer/match/_/gameId/760433) | 06-17 01:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 20 | [Austria – Jordan](https://www.espn.com/soccer/match/_/gameId/760431) | 06-17 04:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G1 | 21 | [Portugal – Congo DR](https://www.espn.com/soccer/match/_/gameId/760435) | 06-17 17:00 | Before | C · CSV | P · W/D/L + five |
| G1 | 22 | [England – Croatia](https://www.espn.com/soccer/match/_/gameId/760437) | 06-17 20:00 | Before | C · CSV | P · W/D/L + five |
| G1 | 23 | [Ghana – Panama](https://www.espn.com/soccer/match/_/gameId/760434) | 06-17 23:00 | Before | C · CSV | P · W/D/L + five |
| G1 | 24 | [Uzbekistan – Colombia](https://www.espn.com/soccer/match/_/gameId/760436) | 06-18 02:00 | Before | C · CSV | P · W/D/L + five |
| G2 | 25 | [Czechia – South Africa](https://www.espn.com/soccer/match/_/gameId/760438) | 06-18 16:00 | After | B · W/D/L + five | P · W/D/L + five |
| G2 | 26 | [Switzerland – Bosnia-Herzegovina](https://www.espn.com/soccer/match/_/gameId/760439) | 06-18 19:00 | After | — | — |
| G2 | 27 | [Canada – Qatar](https://www.espn.com/soccer/match/_/gameId/760440) | 06-18 22:00 | After | — | — |
| G2 | 28 | [Mexico – South Korea](https://www.espn.com/soccer/match/_/gameId/760441) | 06-19 01:00 | After | — | — |
| G2 | 29 | [United States – Australia](https://www.espn.com/soccer/match/_/gameId/760442) | 06-19 19:00 | After | B · W/D/L + five | P · W/D/L + five |
| G2 | 30 | [Scotland – Morocco](https://www.espn.com/soccer/match/_/gameId/760445) | 06-19 22:00 | After | — | — |
| G2 | 31 | [Brazil – Haiti](https://www.espn.com/soccer/match/_/gameId/760444) | 06-20 00:30 | After | A · W/D/L | L · W/D/L |
| G2 | 32 | [Türkiye – Paraguay](https://www.espn.com/soccer/match/_/gameId/760443) | 06-20 03:00 | After | B · W/D/L + five | P · W/D/L + five |
| G2 | 33 | [Netherlands – Sweden](https://www.espn.com/soccer/match/_/gameId/760447) | 06-20 17:00 | After | — | — |
| G2 | 34 | [Germany – Ivory Coast](https://www.espn.com/soccer/match/_/gameId/760448) | 06-20 20:00 | After | A · W/D/L | L · W/D/L |
| G2 | 35 | [Ecuador – Curaçao](https://www.espn.com/soccer/match/_/gameId/760446) | 06-21 00:00 | After | A · W/D/L | L · W/D/L |
| G2 | 36 | [Tunisia – Japan](https://www.espn.com/soccer/match/_/gameId/760449) | 06-21 04:00 | After | — | — |
| G2 | 37 | [Spain – Saudi Arabia](https://www.espn.com/soccer/match/_/gameId/760453) | 06-21 16:00 | After | D · W/D/L + five | — |
| G2 | 38 | [Belgium – Iran](https://www.espn.com/soccer/match/_/gameId/760451) | 06-21 19:00 | After | D · W/D/L + five | — |
| G2 | 39 | [Uruguay – Cape Verde](https://www.espn.com/soccer/match/_/gameId/760450) | 06-21 22:00 | After | D · W/D/L + five | — |
| G2 | 40 | [New Zealand – Egypt](https://www.espn.com/soccer/match/_/gameId/760452) | 06-22 01:00 | After | D · W/D/L + five | — |
| G2 | 41 | [Argentina – Austria](https://www.espn.com/soccer/match/_/gameId/760456) | 06-22 17:00 | After | D · W/D/L + five | L · W/D/L |
| G2 | 42 | [France – Iraq](https://www.espn.com/soccer/match/_/gameId/760457) | 06-22 21:00 | After | D · W/D/L + five | — |
| G2 | 43 | [Norway – Senegal](https://www.espn.com/soccer/match/_/gameId/760454) | 06-23 00:00 | After | D · W/D/L + five | — |
| G2 | 44 | [Jordan – Algeria](https://www.espn.com/soccer/match/_/gameId/760455) | 06-23 03:00 | After | D · W/D/L + five | L · W/D/L |
| G2 | 45 | [Portugal – Uzbekistan](https://www.espn.com/soccer/match/_/gameId/760461) | 06-23 17:00 | After | D · W/D/L + five | L · W/D/L |
| G2 | 46 | [England – Ghana](https://www.espn.com/soccer/match/_/gameId/760458) | 06-23 20:00 | After | D · W/D/L + five | L · W/D/L |
| G2 | 47 | [Panama – Croatia](https://www.espn.com/soccer/match/_/gameId/760460) | 06-23 23:00 | After | D · W/D/L + five | L · W/D/L |
| G2 | 48 | [Colombia – Congo DR](https://www.espn.com/soccer/match/_/gameId/760459) | 06-24 02:00 | After | D · W/D/L + five | L · W/D/L |
| G3 | 49 | [Switzerland – Canada](https://www.espn.com/soccer/match/_/gameId/760463) | 06-24 19:00 | Before | — | — |
| G3 | 50 | [Bosnia-Herzegovina – Qatar](https://www.espn.com/soccer/match/_/gameId/760462) | 06-24 19:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G3 | 51 | [Scotland – Brazil](https://www.espn.com/soccer/match/_/gameId/760465) | 06-24 22:00 | Before | A · W/D/L | L · W/D/L |
| G3 | 52 | [Morocco – Haiti](https://www.espn.com/soccer/match/_/gameId/760464) | 06-24 22:00 | Before | — | — |
| G3 | 53 | [Czechia – Mexico](https://www.espn.com/soccer/match/_/gameId/760467) | 06-25 01:00 | Before | — | — |
| G3 | 54 | [South Africa – South Korea](https://www.espn.com/soccer/match/_/gameId/760466) | 06-25 01:00 | Before | A · W/D/L | L · W/D/L |
| G3 | 55 | [Curaçao – Ivory Coast](https://www.espn.com/soccer/match/_/gameId/760473) | 06-25 20:00 | Before | — | — |
| G3 | 56 | [Ecuador – Germany](https://www.espn.com/soccer/match/_/gameId/760468) | 06-25 20:00 | Before | — | — |
| G3 | 57 | [Japan – Sweden](https://www.espn.com/soccer/match/_/gameId/760471) | 06-25 23:00 | Before | — | — |
| G3 | 58 | [Tunisia – Netherlands](https://www.espn.com/soccer/match/_/gameId/760472) | 06-25 23:00 | Before | — | — |
| G3 | 59 | [Türkiye – United States](https://www.espn.com/soccer/match/_/gameId/760470) | 06-26 02:00 | Before | — | — |
| G3 | 60 | [Paraguay – Australia](https://www.espn.com/soccer/match/_/gameId/760469) | 06-26 02:00 | Before | B · W/D/L + five | P · W/D/L + five |
| G3 | 61 | [Norway – France](https://www.espn.com/soccer/match/_/gameId/760475) | 06-26 19:00 | Before | — | — |
| G3 | 62 | [Senegal – Iraq](https://www.espn.com/soccer/match/_/gameId/760474) | 06-26 19:00 | Before | A · W/D/L | L · W/D/L |
| G3 | 63 | [Cape Verde – Saudi Arabia](https://www.espn.com/soccer/match/_/gameId/760478) | 06-27 00:00 | Before | — | — |
| G3 | 64 | [Uruguay – Spain](https://www.espn.com/soccer/match/_/gameId/760479) | 06-27 00:00 | Before | — | — |
| G3 | 65 | [Egypt – Iran](https://www.espn.com/soccer/match/_/gameId/760476) | 06-27 03:00 | Before | — | — |
| G3 | 66 | [New Zealand – Belgium](https://www.espn.com/soccer/match/_/gameId/760477) | 06-27 03:00 | Before | — | — |
| G3 | 67 | [Panama – England](https://www.espn.com/soccer/match/_/gameId/760485) | 06-27 21:00 | Before | — | — |
| G3 | 68 | [Croatia – Ghana](https://www.espn.com/soccer/match/_/gameId/760480) | 06-27 21:00 | Before | — | — |
| G3 | 69 | [Colombia – Portugal](https://www.espn.com/soccer/match/_/gameId/760481) | 06-27 23:30 | Before | — | — |
| G3 | 70 | [Congo DR – Uzbekistan](https://www.espn.com/soccer/match/_/gameId/760482) | 06-27 23:30 | Before | — | — |
| G3 | 71 | [Algeria – Austria](https://www.espn.com/soccer/match/_/gameId/760484) | 06-28 02:00 | Before | — | — |
| G3 | 72 | [Jordan – Argentina](https://www.espn.com/soccer/match/_/gameId/760483) | 06-28 02:00 | Before | — | — |

</details>

<details>
<summary>Knockout stage (32 matches)</summary>

| Stage | CSV match_id | Fixture (ESPN) | Kickoff (UTC, 2026) | Retained CSV export | GitHub before kickoff |
|---|---:|---|---|---|---|
| R32 | 760486 | [South Africa – Canada](https://www.espn.com/soccer/match/_/gameId/760486) | 06-28 19:00 | Before | — |
| R32 | 760487 | [Brazil – Japan](https://www.espn.com/soccer/match/_/gameId/760487) | 06-29 17:00 | Before | — |
| R32 | 760489 | [Germany – Paraguay](https://www.espn.com/soccer/match/_/gameId/760489) | 06-29 20:30 | Before | — |
| R32 | 760488 | [Netherlands – Morocco](https://www.espn.com/soccer/match/_/gameId/760488) | 06-30 01:00 | Before | — |
| R32 | 760490 | [Ivory Coast – Norway](https://www.espn.com/soccer/match/_/gameId/760490) | 06-30 17:00 | Before | — |
| R32 | 760492 | [France – Sweden](https://www.espn.com/soccer/match/_/gameId/760492) | 06-30 21:00 | Before | — |
| R32 | 760491 | [Mexico – Ecuador](https://www.espn.com/soccer/match/_/gameId/760491) | 07-01 02:00 | Before | — |
| R32 | 760495 | [England – Congo DR](https://www.espn.com/soccer/match/_/gameId/760495) | 07-01 16:00 | Before | — |
| R32 | 760493 | [Belgium – Senegal](https://www.espn.com/soccer/match/_/gameId/760493) | 07-01 20:00 | Before | — |
| R32 | 760494 | [United States – Bosnia-Herzegovina](https://www.espn.com/soccer/match/_/gameId/760494) | 07-02 00:00 | Before | — |
| R32 | 760497 | [Spain – Austria](https://www.espn.com/soccer/match/_/gameId/760497) | 07-02 19:00 | Before | — |
| R32 | 760496 | [Portugal – Croatia](https://www.espn.com/soccer/match/_/gameId/760496) | 07-02 23:00 | Before | — |
| R32 | 760498 | [Switzerland – Algeria](https://www.espn.com/soccer/match/_/gameId/760498) | 07-03 03:00 | Before | — |
| R32 | 760499 | [Australia – Egypt](https://www.espn.com/soccer/match/_/gameId/760499) | 07-03 18:00 | Before | — |
| R32 | 760500 | [Argentina – Cape Verde](https://www.espn.com/soccer/match/_/gameId/760500) | 07-03 22:00 | Before | — |
| R32 | 760501 | [Colombia – Ghana](https://www.espn.com/soccer/match/_/gameId/760501) | 07-04 01:30 | Before | — |
| R16 | 760502 | [Canada – Morocco](https://www.espn.com/soccer/match/_/gameId/760502) | 07-04 17:00 | Before | E · CSV |
| R16 | 760503 | [Paraguay – France](https://www.espn.com/soccer/match/_/gameId/760503) | 07-04 21:00 | Before | E · CSV |
| R16 | 760504 | [Brazil – Norway](https://www.espn.com/soccer/match/_/gameId/760504) | 07-05 20:00 | Before | E · CSV |
| R16 | 760505 | [Mexico – England](https://www.espn.com/soccer/match/_/gameId/760505) | 07-06 01:00 | Before | E · CSV |
| R16 | 760506 | [Portugal – Spain](https://www.espn.com/soccer/match/_/gameId/760506) | 07-06 19:00 | Before | E · CSV |
| R16 | 760507 | [United States – Belgium](https://www.espn.com/soccer/match/_/gameId/760507) | 07-07 00:00 | Before | E · CSV |
| R16 | 760509 | [Argentina – Egypt](https://www.espn.com/soccer/match/_/gameId/760509) | 07-07 16:00 | Before | E · CSV |
| R16 | 760508 | [Switzerland – Colombia](https://www.espn.com/soccer/match/_/gameId/760508) | 07-07 20:00 | Before | E · CSV |
| QF | 760510 | [France – Morocco](https://www.espn.com/soccer/match/_/gameId/760510) | 07-09 20:00 | After | — |
| QF | 760511 | [Spain – Belgium](https://www.espn.com/soccer/match/_/gameId/760511) | 07-10 19:00 | After | — |
| QF | 760512 | [Norway – England](https://www.espn.com/soccer/match/_/gameId/760512) | 07-11 21:00 | After | — |
| QF | 760513 | [Argentina – Switzerland](https://www.espn.com/soccer/match/_/gameId/760513) | 07-12 01:00 | After | — |
| SF | 760514 | [France – Spain](https://www.espn.com/soccer/match/_/gameId/760514) | 07-14 19:00 | Before | — |
| SF | 760515 | [England – Argentina](https://www.espn.com/soccer/match/_/gameId/760515) | 07-15 19:00 | Before | — |
| Third place | 760516 | [France – England](https://www.espn.com/soccer/match/_/gameId/760516) | 07-18 21:00 | Before | — |
| Final | 760517 | [Spain – Argentina](https://www.espn.com/soccer/match/_/gameId/760517) | 07-19 19:00 | Before | — |

</details>

## Source paths and checksums

The nine CSVs at the [README revision](https://github.com/KurokawaMasaya/fifa_world_cup_2026/tree/c8a98448494568c787e7733b4d18ca5aef5030ef/output) match the corresponding entries in the original project archive, byte for byte. Paths below start at the repository root, or at `FIFAproject2026/` inside the ZIP. The checksums identify content. The timing evidence comes from the separate records above.

<details>
<summary>Original metadata paths and SHA-256</summary>

Each JSON sits beside its stage CSV. Linked metadata has a byte-identical historical GitHub copy. The other JSON files and the final fixture log are identified by their original archive paths and hashes.

| Source | Path in original archive | SHA-256 |
|---|---|---|
| G1 | [`output/predictions/live/group_stage_round1_predictions__v24_abcd_no_e__metadata.json`](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/9400cfd610040ef3706ba636fb3cc6fd015f590f/output/predictions/live/group_stage_round1_predictions__v24_abcd_no_e__metadata.json) | `0dfddf9d498aeef9ec077456ad9be2ba728a3d359a50feb88a7d73dcff49a704` |
| G2 | `output/predictions/live/group_stage_round2_predictions__metadata.json` | `90f272d74ce99985508948dd672095d4455c04c7dbd84b0046dee680424861e0` |
| G3 | `output/predictions/live/group_stage_round3_predictions__metadata.json` | `68283812f4da32bc56ce370647e24961f605e399daeb45819b25ea202d7c5aa8` |
| R32 | [`output/predictions/round_of_32_predictions_after_group_stage_clean_with_et_penalties__metadata.json`](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c21c729f54eb5130509dffed8cbf0ec5c2a3254e/output/predictions/round_of_32_predictions_after_group_stage_clean_with_et_penalties__metadata.json) | `f5c33f0f9b5c6ac5e83dffa527c361bcdc985066ab4856730f7667f538536c05` |
| R16 | [`output/predictions/round_of_16_predictions_after_r32_clean__metadata.json`](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/c21c729f54eb5130509dffed8cbf0ec5c2a3254e/output/predictions/round_of_16_predictions_after_r32_clean__metadata.json) | `e6e10b484257fe2083edba65f66847fae05d58b65e9fbc5151c4a3e3ae94c010` |
| QF | `output/predictions/R8_predictions_clean__metadata.json` | `e89cf4a0f18c7a12bb036107a3e49b5f3c791450857f36dc2712fcaec4fa22f6` |
| SF | `output/predictions/semifinal_predictions_clean__metadata.json` | `3b97be33a068ab54e7b48a87213c75025e0017c30a9aef2dc11dafac91eab9f6` |
| F | `output/predictions/third_place_and_final_predictions_clean__metadata.json` | `17ec96adc1b57f0a044fc11e636db4165ef423039fcdc3c7f051e5dd2d02a00a` |
| P · earlier CSV | [`output/predictions/group_stage_predictions_clean.csv`](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/cd339c99dd80024c2c39fe12468fd3f8dbd5d9e1/output/predictions/group_stage_predictions_clean.csv) | `28220f53467a0333b0a45b94ce8a8c22a2bedbcc5ba576a0eb4ca036e2326502` |
| P · generation | [`output/predictions/group_stage_predictions_clean__metadata.json`](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/cd339c99dd80024c2c39fe12468fd3f8dbd5d9e1/output/predictions/group_stage_predictions_clean__metadata.json) | `101d429de6e615ff6566d7eb7e0ee19855f946f4e062af3c30044ec53788f045` |
| L · earlier CSV | [`legacy/outputs/predictions/legacy_prediction_versions/group_stage_predictions_v2_latest.csv`](https://github.com/KurokawaMasaya/fifa_world_cup_2026/blob/6b29e0b4330dc7ee7f4dae7d10f74de7fa692677/output/predictions/group_stage_predictions_v2_latest.csv) | `45c3ca269217357ce2046938954be612aa6082f7d8d2449cedd3718a844d02e7` |
| Final fixture log | `output/live/fixtures_results.csv` | `e7a27b569dc5f2a77fb6ccb94665ec0b11287db2f25629ebf91e7b5d53fc69a9` |

Original `FIFAproject2026.zip` SHA-256: `1a7fd30d06dff4ddd027ff65c8ee7b76e9b0c831d5f7b006b92096280a60f39f`.

</details>

<details>
<summary>SHA-256 manifest for the nine published CSVs</summary>

Save the following block as `forecast.sha256` and run `shasum -a 256 -c forecast.sha256` from the repository root.

```text
eee5a0fb862b3d515a6e70c0a4aa9c2bc201d3f5cdd5f46e6af497f1c5977607  output/predictions/live/group_stage_round1_predictions__v24_abcd_no_e.csv
d6ce4cb081d2f843b97b222e06369974e63bedfc23a3ab6158460f8c97604f98  output/predictions/live/group_stage_round2_predictions_clean.csv
4240d4274e54dc2ab4a8153a082385478be278aa9c8f7f8f36cd636fd2c2896d  output/predictions/live/group_stage_round3_predictions_clean.csv
5125bc7047f87cc2d108c92020f764b8d7ef35abe1fc4ae0965865bf18b6c4cf  output/predictions/round_of_32_predictions_after_group_stage_clean_with_et_penalties.csv
e71c89772be8f1a10d482df3aac9167c9f0621711ce3b74014f75b227b6632e6  output/predictions/round_of_16_predictions_after_r32_clean.csv
4f2dae3e5e4fde2803ae0465b3a82f07aa91bf571982808914ce61c51ccde814  output/predictions/R8_predictions_clean.csv
4624e0d71e0117c9deb76e0ca47aac8a3f2617dd955f48bcae33331f9f2cfc18  output/predictions/semifinal_predictions_clean.csv
7fab279e8b85e965fab36e61f112826ab9864b460751dfeae665b0493baf7075  output/predictions/third_place_and_final_predictions_clean.csv
81ea5c4df53838eb7edb81202896d334ddb8d851090fc12cea08592ab10b640b  output/simulations/tournament_simulation_results_default.csv
```

</details>
