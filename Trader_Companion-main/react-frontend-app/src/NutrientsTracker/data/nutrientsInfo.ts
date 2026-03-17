export const nutrientInfo: Record<string, { benefits: string; deficiency: string; prosCons: string }> = {
  // Macronutrients
  "Protein": {
    benefits: "Essential for muscle repair, growth, immune function, and enzymes.",
    deficiency: "Muscle wasting, weakened immunity, hair/nail thinning.",
    prosCons: "Pro: High satiety, builds tissue. Con: Very high amounts strain kidneys if dehydrated."
  },
  "Carbohydrates": {
    benefits: "Primary energy source for the brain and high-intensity exercise.",
    deficiency: "Fatigue, brain fog, hormonal imbalances (in extreme restriction).",
    prosCons: "Pro: Fast energy, mood regulation. Con: Excess refined carbs trigger fat storage and insulin spikes."
  },
  "Fiber": {
    benefits: "Feeds microbiome, regulates blood sugar, and optimizes digestion.",
    deficiency: "Constipation, poor gut health, spiked blood sugar.",
    prosCons: "Pro: Controls cholesterol, highly satiating. Con: Too much too quickly causes bloating/gas."
  },
  "Fats (Total)": {
    benefits: "Hormone production, cell membrane health, absorbs A/D/E/K vitamins.",
    deficiency: "Hormonal crash, poor brain function, dry skin.",
    prosCons: "Pro: Sustained energy, rich flavor. Con: Very calorie dense, easy to over-consume."
  },
  "Omega-3": {
    benefits: "Potent anti-inflammatory, critical for brain/eye health, lowers triglycerides.",
    deficiency: "Brain fog, joint pain, dry eyes/skin, higher inflammation.",
    prosCons: "Pro: Heart protective. Con: High doses thin blood; rancid supplements cause inflammation."
  },
  "Omega-6": {
    benefits: "Structural integrity of cells, necessary for immune responses.",
    deficiency: "Skin lesions, poor wound healing (rare).",
    prosCons: "Pro: Essential for life. Con: Highly inflammatory if heavily outweighed vs Omega-3."
  },
  "Water": {
    benefits: "Lubricates joints, temperature regulation, flushes toxins.",
    deficiency: "Dehydration, brain fog, cramping, fatal if completely restricted.",
    prosCons: "Pro: Calorie-free vitality. Con: Excessive amounts (gallons) can flush essential electrolytes (hyponatremia)."
  },

  // Fat-Soluble
  "Vitamin A": {
    benefits: "Crucial for vision, immune system, and cellular differentiation.",
    deficiency: "Night blindness, hyperkeratosis (bumpy skin), increased infection risk.",
    prosCons: "Pro: Anti-aging (retinol), repairs tissues. Con: Highly toxic in massive doses (liver damage)."
  },
  "Vitamin D3": {
    benefits: "Calcium absorption, immune modulation, testosterone/hormone synthesis.",
    deficiency: "Bone loss (osteopenia), depression, low immunity.",
    prosCons: "Pro: Essentially a pro-hormone. Con: Toxicity causes hypercalcemia (calcium in blood)."
  },
  "Vitamin E": {
    benefits: "Powerful antioxidant, protects cell membranes from oxidative stress.",
    deficiency: "Nerve damage, muscle weakness, vision problems.",
    prosCons: "Pro: Protects skin/eyes. Con: High supplementary doses linked to bleeding."
  },
  "Vitamin K1": {
    benefits: "Activates proteins required for blood clotting.",
    deficiency: "Excessive bleeding, bruising easily.",
    prosCons: "Pro: Readily available in greens. Con: Interferes with blood thinners like Warfarin."
  },
  "Vitamin K2": {
    benefits: "Directs calcium INTO bones/teeth and AWAY from soft tissues/arteries.",
    deficiency: "Arterial calcification, osteoporosis, dental cavities.",
    prosCons: "Pro: Prevents heart disease. Con: Hard to get from standard Western diet."
  },

  // Water-Soluble
  "B1 (Thiamin)": {
    benefits: "Converts carbs into energy, brain and nerve function.",
    deficiency: "Beriberi, Wernicke-Korsakoff syndrome (memory loss).",
    prosCons: "Pro: Essential for metabolism. Con: Easily depleted by alcohol/sugar."
  },
  "B2 (Riboflavin)": {
    benefits: "Energy production, cellular function, and metabolism of fats/drugs.",
    deficiency: "Cracked lips, sore throat, bloodshot eyes.",
    prosCons: "Pro: Migraine prevention. Con: Turns urine neon yellow (harmless)."
  },
  "B3 (Niacin)": {
    benefits: "DNA repair, boosts HDL cholesterol, energy transfer.",
    deficiency: "Pellagra (dermatitis, diarrhea, dementia).",
    prosCons: "Pro: Lowers bad cholesterol. Con: High doses cause severe 'niacin flush' (red/itchy skin)."
  },
  "B5 (Pantothenic Acid)": {
    benefits: "Synthesizes coenzyme A, builds hormones/cholesterol.",
    deficiency: "Fatigue, 'burning feet' syndrome.",
    prosCons: "Pro: Broadly available in food. Con: Practically impossible to overdose."
  },
  "B6 (Pyridoxine)": {
    benefits: "Amino acid metabolism, creates neurotransmitters (serotonin, dopamine).",
    deficiency: "Depression, confusion, weakened immunity.",
    prosCons: "Pro: Enhances mood/sleep. Con: Chronic mega-doses cause permanent nerve damage (neuropathy)."
  },
  "B7 (Biotin)": {
    benefits: "Metabolizes fats/carbs, strengthens hair, skin, and nails.",
    deficiency: "Hair loss, scaly red skin rash.",
    prosCons: "Pro: Great for cosmetics/skin. Con: High doses famously skew blood test results (e.g., thyroid)."
  },
  "B9 (Folate)": {
    benefits: "DNA synthesis, red blood cell formation, critical for pregnancy.",
    deficiency: "Megaloblastic anemia, neural tube defects in fetuses.",
    prosCons: "Pro: Prevents birth defects. Con: Synthetic folic acid can mask B12 deficiency."
  },
  "B12 (Cobalamin)": {
    benefits: "Nerve myelin sheath maintenance, DNA and red blood cell production.",
    deficiency: "Irreversible nerve damage, severe fatigue, megaloblastic anemia.",
    prosCons: "Pro: Massive energy boost if deficient. Con: Only found in animal products (vegans must supplement)."
  },
  "Vitamin C": {
    benefits: "Collagen synthesis, potent antioxidant, boosts immune function.",
    deficiency: "Scurvy (bleeding gums, lost teeth, unhealed wounds).",
    prosCons: "Pro: Highly safe, flushes out. Con: Massive doses can cause diarrhea or kidney stones."
  },
  "Choline": {
    benefits: "Liver function, brain development, creates acetylcholine (memory).",
    deficiency: "Fatty liver disease, muscle damage, cognitive decline.",
    prosCons: "Pro: Potent nootropic/memory booster. Con: Hard to hit RDA without eating egg yolks/liver."
  },

  // Macrominerals
  "Calcium": {
    benefits: "Bone/teeth structure, muscle contractions, nerve signaling.",
    deficiency: "Osteoporosis, muscle cramps (tetany).",
    prosCons: "Pro: Dense bones. Con: If taken without D3/K2/Magnesium, calcifies arteries instead of bones."
  },
  "Phosphorus": {
    benefits: "Forms ATP (energy), bone matrix, cell membranes.",
    deficiency: "Muscle weakness, bone pain.",
    prosCons: "Pro: Ubiquitous in food. Con: Excess (often from soda) leaches calcium from bones."
  },
  "Magnesium": {
    benefits: "Over 300 enzyme reactions, muscle relaxation, sleep, nervous system.",
    deficiency: "Twitches, cramps, anxiety, insomnia, palpitations.",
    prosCons: "Pro: Ultimate relaxation mineral. Con: Cheap versions (Oxide) act only as powerful laxatives."
  },
  "Sodium": {
    benefits: "Fluid balance, nerve impulses, muscle contractions.",
    deficiency: "Hyponatremia (cramps, nausea, coma in severe cases).",
    prosCons: "Pro: Vital for athletic endurance/hydration. Con: Excess raises blood pressure in sensitive individuals."
  },
  "Potassium": {
    benefits: "Counterbalances Sodium, lowers blood pressure, intracellular fluid balance.",
    deficiency: "Severe cramping, heart arrhythmias, fatigue.",
    prosCons: "Pro: Highly protective of cardiovascular health. Con: Supplemental limits are strictly capped due to heart risk."
  },
  "Chloride": {
    benefits: "Stomach acid production (HCl), fluid balance.",
    deficiency: "Alkalosis, poor digestion.",
    prosCons: "Pro: Easy to obtain (table salt). Con: Rarely deficient."
  },

  // Trace Minerals
  "Iron": {
    benefits: "Creates Hemoglobin to carry oxygen in the blood.",
    deficiency: "Anemia, extreme fatigue, pale skin, shortness of breath.",
    prosCons: "Pro: Essential for vitality. Con: Highly toxic if over-accumulated (men/post-menopausal women shouldn't supplement without need)."
  },
  "Zinc": {
    benefits: "Testosterone production, immune system master, wound healing.",
    deficiency: "Loss of taste/smell, delayed healing, hypogonadism.",
    prosCons: "Pro: Blocks cold viruses from replicating. Con: Depletes Copper if taken long-term."
  },
  "Iodine": {
    benefits: "Synthesizes thyroid hormones (T3/T4) controlling metabolism.",
    deficiency: "Goiter (enlarged thyroid), hypothyroidism, severe metabolic slowdown.",
    prosCons: "Pro: Keeps thyroid firing. Con: Excess can trigger autoimmune thyroid issues (Hashimoto’s)."
  },
  "Selenium": {
    benefits: "Activates thyroid hormones, potent antioxidant.",
    deficiency: "Keshan disease (heart damage), poor thyroid function.",
    prosCons: "Pro: Just 2 Brazil nuts hit the RDA. Con: Very narrow safety window; toxic if overdosed."
  },
  "Copper": {
    benefits: "Energy production, iron metabolism, melanin formulation.",
    deficiency: "Anemia (resembles iron deficiency), gray hair, neuropathy.",
    prosCons: "Pro: Balances zinc. Con: Excess stored in liver/brain causing toxicity (Wilson's disease)."
  },
  "Manganese": {
    benefits: "Bone formation, antioxidant defense (SOD).",
    deficiency: "Impaired growth, skeletal abnormalities.",
    prosCons: "Pro: Required for healthy joints. Con: Rare deficiency, toxic if inhaled (industrial fumes)."
  },

  // Amino Acids
  "Glycine": {
    benefits: "Crucial for collagen synthesis, joint repair, sleep quality, and balancing methionine from muscle meats.",
    deficiency: "Poor sleep, weak connective tissue, joint pain.",
    prosCons: "Pro: Amazing for sleep and skin/joints. Con: Hard to get enough without bone broth or supplementation."
  },

  // Phytonutrients
  "Lutein": {
    benefits: "Critical for eye health, filters harmful blue light, and protects against macular degeneration.",
    deficiency: "Increased risk of age-related eye damage and oxidative stress in the retina.",
    prosCons: "Pro: Potent eye protector. Con: Best absorbed when consumed with healthy fats."
  },
  "Zeaxanthin": {
    benefits: "Concentrates in the center of the retina to protect against UV damage and oxidative stress.",
    deficiency: "Macular pigment thinning, increased light sensitivity.",
    prosCons: "Pro: Synergistic with Lutein. Con: Rare in Western diets aside from corn/orange peppers."
  },
  "Anthocyanins": {
    benefits: "Potent brain-protective antioxidants, improves vascular health and insulin sensitivity.",
    deficiency: "Reduced antioxidant capacity, higher risk of inflammation-related decline.",
    prosCons: "Pro: Strong anti-aging effects for brain/heart. Con: Easily destroyed by high-heat cooking."
  },
  "Lycopene": {
    benefits: "Powerful antioxidant that protects against UV damage (internal sunscreen) and supports prostate/heart health.",
    deficiency: "Increased susceptibility to oxidative stress and cellular damage.",
    prosCons: "Pro: Highly bioavailable when cooked (e.g., tomato paste). Con: Mostly limited to red/pink fruits and veggies."
  },
  "EGCG (Catechins)": {
    benefits: "High thermogenic effect (fat burning), neuroprotective, and excellent for cardiovascular health.",
    deficiency: "Lower metabolic efficiency and reduced antioxidant defenses.",
    prosCons: "Pro: Found abundantly in green tea/matcha. Con: Can interfere with iron absorption if taken with meals."
  },
};
