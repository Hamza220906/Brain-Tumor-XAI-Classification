TREATMENT_GUIDELINES = {
    'glioma': {
        'urgency': '🔴 CRITICAL',
        'protocol': 'Surgical resection + Radiotherapy & Temozolomide (Stupp Protocol)',
        'follow_up': 'MRI every 3 months for first 2 years'
    },
    'meningioma': {
        'urgency': '🟡 MODERATE',
        'protocol': 'Observation if asymptomatic; Surgical excision or Stereotactic Radiosurgery if symptomatic',
        'follow_up': 'Annual MRI for 5 years'
    },
    'pituitary': {
        'urgency': ' STANDARD',
        'protocol': 'Transsphenoidal surgery; Dopamine agonists (Cabergoline) for prolactinomas',
        'follow_up': 'Hormone panel test + MRI in 6 months'
    },
    'notumor': {
        'urgency': '🔵 ROUTINE',
        'protocol': 'No intervention required. Clinical correlation recommended if symptoms persist.',
        'follow_up': 'Repeat MRI in 6-12 months if neurological symptoms present'
    }
}

DISCLAIMER = "⚠️ AI-generated suggestion based on NCCN guidelines. Final decision must be made by a qualified neuro-oncologist."