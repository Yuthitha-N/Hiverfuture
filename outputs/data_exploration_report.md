# Data Exploration & Brand Selection Report: AppleSupport

## 1. Dataset Overview
- **Source**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)
- **Total Dataset Size**: 794,335 multi-turn conversation threads across dozens of international brands.
- **AppleSupport Total Volume**: 76,639 conversations.

## 2. Brand Selection Justification: Why AppleSupport?
1. **High Interaction Volume**: AppleSupport is the #2 most active brand in the dataset (76,639 conversations), guaranteeing ample data for retrieval and evaluation without data sparsity.
2. **High Language Quality**: Over 95% of customer interactions are in clear English with well-formed problem descriptions (e.g. device model names, iOS versions).
3. **Diverse & Grounded Technical Intents**: Covers software updates, battery health, Apple ID security, App Store billing, iCloud syncing, and hardware issues.
4. **Realistic Escalation Boundaries**: Technical issues can be automated with troubleshooting workflows, while sensitive issues (Apple ID lockout, unauthorized charges, hardware repair) require deterministic human escalation.

## 3. Data Cleaning & Preprocessing Summary
- **Mentions Stripped**: Redacted Twitter handles (`@AppleSupport`, `@115858`) to prevent keyword bias.
- **URL Normalization**: Converted short links (`https://t.co/...`) to `[URL]` tokens.
- **Deduplication**: Normalized prefix matching removed duplicate retweets and bot spam.
- **Turn Parsing**: Separated inbound initial customer problem statements from agent resolutions.

## 4. Intent Breakdown
| Intent | Count | Percentage |
|---|---|---|
| `ios_software_update` | 5,724 | 57.2% |
| `general_inquiry_features` | 2,502 | 25.0% |
| `battery_performance` | 796 | 8.0% |
| `device_hardware_display` | 291 | 2.9% |
| `connectivity_network_bluetooth` | 272 | 2.7% |
| `data_sync_backup_icloud` | 176 | 1.8% |
| `apple_id_account_security` | 157 | 1.6% |
| `app_store_billing_subscriptions` | 82 | 0.8% |

## 5. Text Statistics
- **Avg Customer Message Length**: 18.1 words
- **Avg Support Response Length**: 19.71 words
- **Clean Subsample Size**: 10,000 pairs (8,000 train/retrieval, 2,000 validation)
