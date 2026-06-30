---
name: query-wikipedia
description: Search Wikipedia and return a summary of the topic.
parameters: {"type": "object", "properties": {"query": {"type": "string", "description": "The search query for Wikipedia"}}}
---

Please use this skill to query wikipedia for a topic. Pass the query as JSON:
```json
{
  "query": "Albert Einstein"
}
```
