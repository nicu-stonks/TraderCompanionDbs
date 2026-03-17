import { Progress } from '@/components/ui/progress';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { nutrientInfo } from '../data/nutrientsInfo';

export const nutrientCategories = {
  "Macronutrients": [
    "Protein", "Carbohydrates", "Fiber", "Fats (Total)", "Omega-3", "Omega-6", "Water"
  ],
  "Fat-Soluble Vitamins": [
    "Vitamin A", "Vitamin D3", "Vitamin E", "Vitamin K1", "Vitamin K2"
  ],
  "Water-Soluble Vitamins": [
    "B1 (Thiamin)", "B2 (Riboflavin)", "B3 (Niacin)", "B5 (Pantothenic Acid)",
    "B6 (Pyridoxine)", "B7 (Biotin)", "B9 (Folate)", "B12 (Cobalamin)", "Vitamin C", "Choline"
  ],
  "Macrominerals": [
    "Calcium", "Phosphorus", "Magnesium", "Sodium", "Potassium", "Chloride"
  ],
  "Trace Minerals": [
    "Iron", "Zinc", "Iodine", "Selenium", "Copper", "Manganese"
  ],
  "Amino Acids": [
    "Glycine"
  ],
  "Phytonutrients": [
    "Lutein", "Zeaxanthin", "Anthocyanins", "Lycopene", "EGCG (Catechins)"
  ]
};

const categoryColors: Record<string, string> = {
  "Macronutrients": "#61afef",
  "Fat-Soluble Vitamins": "#98c379",
  "Water-Soluble Vitamins": "#c08aff",
  "Macrominerals": "#d19a66",
  "Trace Minerals": "#be5046",
  "Amino Acids": "#61afef",
  "Phytonutrients": "#e06c75"
};

interface NutrientStatsProps {
  completionData: Record<string, number>;
  sourcesData?: Record<string, string[]>;
}

export function NutrientStats({ completionData, sourcesData }: NutrientStatsProps) {
  // Safe accessor defaulting to 0
  const getVal = (n: string) => completionData[n] || 0;

  const allNutrientsWithCats = Object.entries(nutrientCategories).flatMap(([category, nutrients]) =>
    nutrients.map(n => ({
      name: n,
      category,
      color: categoryColors[category]
    }))
  );

  const totalNutrients = allNutrientsWithCats.length;

  const discreteCompleted = allNutrientsWithCats.filter(n => getVal(n.name) >= 100).length;
  const discretePercentage = Math.round((discreteCompleted / totalNutrients) * 100);

  const totalPercentageSum = allNutrientsWithCats.reduce((sum, n) => sum + Math.min(getVal(n.name), 100), 0);
  const nonDiscretePercentage = Math.round(totalPercentageSum / totalNutrients);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row gap-6 bg-muted/30 p-4 rounded-lg border">
        <div className="flex-1 space-y-2">
          <div className="flex justify-between text-sm font-medium">
            <span>Discrete Completion ({discreteCompleted}/{totalNutrients} fully met)</span>
            <span>{discretePercentage}%</span>
          </div>
          <Progress value={discretePercentage} className="h-3" />
        </div>
        <div className="flex-1 space-y-2">
          <div className="flex justify-between text-sm font-medium">
            <span>Average Completion (capped at 100% per nutrient)</span>
            <span>{nonDiscretePercentage}%</span>
          </div>
          <Progress value={nonDiscretePercentage} className="h-3" />
        </div>
      </div>

      <TooltipProvider delayDuration={150}>
        <div className="columns-1 sm:columns-2 md:columns-3 xl:columns-4 gap-6 pt-2">
          {allNutrientsWithCats
            .slice()
            .sort((a, b) => getVal(a.name) - getVal(b.name))
            .map(({ name, category, color }) => {
              const val = getVal(name);
              const sources = sourcesData?.[name] || [];
              const info = nutrientInfo[name] || {
                benefits: "Data pending.",
                deficiency: "Data pending.",
                prosCons: "Data pending."
              };

              return (
                <Tooltip key={name}>
                  <TooltipTrigger asChild>
                    <div className="space-y-1.5 cursor-help group break-inside-avoid mb-4 block">
                      <div className="flex justify-between text-sm">
                        <span className="font-medium truncate pr-2 group-hover:underline decoration-muted-foreground/50 underline-offset-4">{name}</span>
                        <span className="text-muted-foreground whitespace-nowrap">{val}%</span>
                      </div>
                      {/* Custom inline progress bar to support dynamic hex colors */}
                      <div className="w-full bg-secondary rounded-full h-2 overflow-hidden shadow-inner">
                        <div
                          className="h-full transition-all duration-500 ease-out"
                          style={{
                            width: `${Math.min(val, 100)}%`,
                            backgroundColor: color
                          }}
                        />
                      </div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" className="w-[300px] p-3 shadow-xl z-50">
                    <div className="space-y-2 text-xs leading-relaxed">
                      <div className="font-bold border-b pb-1 mb-1 text-sm bg-clip-text text-transparent" style={{ backgroundImage: `linear-gradient(to right, ${color}, ${color})` }}>
                        {name} <span className="text-muted-foreground font-normal">({category})</span>
                      </div>
                      {sources.length > 0 && (
                        <div className="pb-1 border-b border-muted">
                          <strong className="text-foreground">🍽️ Sources Today:</strong> <span className="text-muted-foreground">{sources.join(", ")}</span>
                        </div>
                      )}
                      <div><strong className="text-foreground">✓ Benefits:</strong> <span className="text-muted-foreground">{info.benefits}</span></div>
                      <div><strong className="text-foreground">⚠ Deficiency:</strong> <span className="text-muted-foreground">{info.deficiency}</span></div>
                      <div><strong className="text-foreground">⚖ Pros/Cons:</strong> <span className="text-muted-foreground">{info.prosCons}</span></div>
                    </div>
                  </TooltipContent>
                </Tooltip>
              );
            })}
        </div>
      </TooltipProvider>
    </div>
  );
}
