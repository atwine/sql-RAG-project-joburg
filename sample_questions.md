# Sample Questions

A progression of questions to try with the RAG system — from simple record lookups to multi-record analysis.  
Run the system with `python main.py` and paste any question at the `>` prompt.

---

## Level 1 — Basic Lookups

Single-record, single-field questions. Good for verifying the system is working.

```
What is the date of birth for Miriam Kirunda?
```
```
What is the other name for Patrick Nansubuga?
```
```
What is the gender of Florence Apio?
```
```
Which village does Charles Lubega live in?
```
```
What is the household relation of Jackline Kirunda?
```
```
What is the Record ID for Stephen Wasswa?
```
```
What is the exit type for John Nalwoga?
```

---

## Level 2 — Record Profiles

Ask for everything known about a specific individual.

```
Tell me everything about the record for Ruth Opio.
```
```
Give me the full profile for Joseph Opio (IND-0008).
```
```
What do we know about Harriet Wasswa?
```
```
Summarise the record for Michael Wasswa (IND-0003).
```

---

## Level 3 — Filtered Lists

Questions that require scanning across multiple records.

```
List all female records in the dataset.
```
```
Which individuals are recorded as HEAD of household?
```
```
Who are the individuals with an exit type of DEATH?
```
```
List all individuals who entered the cohort via BIRTH.
```
```
Which records are still active (no exit type recorded)?
```
```
Who are the individuals living in Namatovu village?
```
```
List all individuals from the Eastern province.
```

---

## Level 4 — Migration & Movement

Questions about why and how people moved.

```
Who migrated for health reasons?
```
```
Which individuals moved due to insecurity?
```
```
Who migrated for education purposes?
```
```
List everyone who had a seasonal migration.
```
```
Which individuals did a return migration back into the cohort?
```
```
Who moved to Kampala?
```
```
Which individuals have an internal migration recorded?
```

---

## Level 5 — Fieldworker & Observation

Questions about data collection and observation events.

```
Which records were collected by Grace Lubega?
```
```
How many individuals were observed during an EXIT event?
```
```
List all records collected by Sarah Nakato.
```
```
Which individuals have a FOLLOW_UP observation type?
```

---

## Level 6 — Comparisons & Analysis

More complex questions requiring the model to reason across several records.

```
How many male versus female individuals are in the dataset?
```
```
Which village has the most records?
```
```
Compare the migration reasons between male and female individuals.
```
```
Who is the oldest individual in the dataset based on date of birth?
```
```
Which individuals share the same surname?
```
```
List all individuals who migrated for marriage.
```
```
How many individuals left the cohort due to OUT_MIGRATION versus DEATH?
```

---

## Level 7 — Open-Ended & Analytical

Questions that require summarisation and interpretation across the full dataset.

```
Give me a summary of migration patterns in this dataset.
```
```
What are the most common reasons people left the cohort?
```
```
Which households appear most frequently across records?
```
```
Are there any individuals who share both a first name and a surname with someone else in the dataset?
```
```
What does the data tell us about migration trends in the Eastern province?
```
```
Identify any individuals who may be from the same family based on shared surnames and villages.
```
