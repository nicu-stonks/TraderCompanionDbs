// components/TradeFilterer.tsx
import React, { useState, useEffect } from 'react'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Trade } from '@/TradeHistoryPage/types/Trade'
import { useFilterOptions } from '../hooks/useFilterOptions'
import type { CustomColumn, CustomColumnValue } from '@/TradeHistoryPage/types/CustomTradeData'
import { customTradeDataAPI } from '@/TradeHistoryPage/services/customTradeDataAPI'

type FilterExtensions = {
  // minEarningsQuality?: number
  // minFundamentalsQuality?: number
  maxPriceTightness?: number
  maxNrBases?: number
  pctOff52WHigh?: number
}

type TradeFiltererProps = {
  filters: Partial<Trade> & FilterExtensions
  onFilterChange: (filters: Partial<Trade> & FilterExtensions) => void
  customColumnFilters?: Record<number, string>  // columnId -> selected value
  onCustomColumnFilterChange?: (filters: Record<number, string>) => void
}

const MIN_VALUE_FIELDS: Record<string, string> = {
  // minEarningsQuality: 'Min Earnings Quality',
  // minFundamentalsQuality: 'Min Fundamentals Quality',
  // maxPriceTightness: 'Max Price Tightness (1 Week Before)',  // Hidden
  // maxNrBases: 'Max Number of Bases',  // Hidden
  // pctOff52WHigh: 'Max % Off 52W High'  // Hidden
}

const DROPDOWN_FIELDS = [
  'Pattern',
  'Status',
  'Category',
  'Market_Condition',
  // 'Exit_Reason',  // Hidden - shown elsewhere as Trade Note
  // 'Has_Earnings_Acceleration',  // Hidden
  // 'Has_Catalyst',
  // 'IPO_Last_10_Years',  // Hidden
  // 'Earnings_Last_Q_20_Pct',
  // 'Volume_Confirmation',
  // 'Is_BioTech',  // Hidden
  // 'Earnings_Surprises',
  // 'Expanding_Margins',
  // 'EPS_breakout',
  // 'Strong_annual_EPS',
  // 'Signs_Acceleration_Will_Continue',
  // 'Sudden_Growth_Change',
  // 'Strong_Quarterly_Sales',
  // 'Strong_Yearly_Sales',
  // 'Positive_Analysts_EPS_Revisions',
  // 'Positive_Analysts_Price_Revisions',
  // 'Ownership_Pct_Change_Past_Earnings',
  // 'Quarters_With_75pct_Surprise',
  // 'Over_10_pct_Avg_Surprise',
  // 'Under_30M_Shares',  // Hidden
  // 'Spikes_On_Volume',
  // 'Started_Off_Correction',
  // 'All_Trendlines_Up',
  // 'If_You_Could_Only_Make_10_Trades',  // Hidden
  'C',
  'A',
  'N',
  'S',
  'L',
  'I',
  'M',
] as const

const formatLabel = (fieldName: string): string => {
  return fieldName
    .split(/(?=[A-Z])|_/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
}

export const TradeFilterer: React.FC<TradeFiltererProps> = ({ filters, onFilterChange, customColumnFilters = {}, onCustomColumnFilterChange }) => {
  const { filterOptions, loading } = useFilterOptions()
  const [customColumns, setCustomColumns] = useState<CustomColumn[]>([])
  const [customValues, setCustomValues] = useState<CustomColumnValue[]>([])

  // Load custom columns and their values for filter options
  useEffect(() => {
    const loadCustomData = async () => {
      try {
        const [colsResp, valsResp] = await Promise.all([
          customTradeDataAPI.getColumns(),
          customTradeDataAPI.getColumnValues(),
        ])
        setCustomColumns(colsResp.data)
        setCustomValues(valsResp.data)
      } catch {
        // silent
      }
    }
    loadCustomData()
  }, [])

  // Build unique values per custom column for dropdowns
  const customColumnOptions: Record<number, string[]> = {}
  customColumns.forEach(col => {
    const vals = customValues
      .filter(v => v.column === col.id && v.value.trim() !== '')
      .map(v => v.value)
    customColumnOptions[col.id] = Array.from(new Set(vals)).sort()
  })

  const handleDropdownChange = (key: keyof Trade, value: string) => {
    let processedValue: Trade[typeof key] | undefined

    switch (typeof filters[key]) {
      case 'boolean':
        processedValue = value === 'true' ? true : value === 'false' ? false : undefined
        break
      case 'number':
        processedValue = value === '' ? undefined : Number(value)
        break
      default:
        processedValue = value === 'any' ? undefined : value
    }

    onFilterChange({ ...filters, [key]: processedValue })
  }

  const handleMinValueChange = (key: keyof FilterExtensions, value: string) => {
    const numValue = value === '' ? undefined : Number(value)
    onFilterChange({ ...filters, [key]: numValue })
  }

  if (loading) return <div>Loading filters...</div>

  const FilterField = ({
    label,
    id,
    children
  }: {
    label: string
    id: string
    children: React.ReactNode
  }) => (
    <div className="flex items-center gap-2">
      <Label htmlFor={id} className="flex-1 text-sm whitespace-nowrap">
        {label}
      </Label>
      <div className="w-48">
        {children}
      </div>
    </div>
  )

  return (
    <div className="grid grid-cols-1 min-[900px]:grid-cols-2 min-[1200px]:grid-cols-3 min-[1600px]:grid-cols-4 min-[2000px]:grid-cols-5 gap-2">
      {/* Numeric input filters */}
      {Object.entries(MIN_VALUE_FIELDS).map(([key, label]) => (
        <FilterField key={key} label={label} id={key}>
          <Input
            id={key}
            type="number"
            min="0"
            value={filters[key as keyof FilterExtensions]?.toString() ?? ''}
            onChange={(e) => handleMinValueChange(
              key as keyof FilterExtensions,
              e.target.value
            )}
          />
        </FilterField>
      ))}

      {/* Dropdown filters */}
      {DROPDOWN_FIELDS.map((fieldName) => {
        const options = filterOptions[fieldName]
        if (!options) return null

        const label = formatLabel(fieldName)
        const isBooleanField = Array.from(options).every(value => typeof value === 'boolean')
        const currentValue = filters[fieldName]?.toString()

        return (
          <FilterField key={fieldName} label={label} id={fieldName}>
            <Select
              value={currentValue || "any"}
              onValueChange={(value) => handleDropdownChange(fieldName, value)}
            >
              <SelectTrigger>
                <SelectValue placeholder="Any" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="any">Any</SelectItem>
                {Array.from(options)
                  .sort((a, b) => a.toString().localeCompare(b.toString()))
                  .map((value) => (
                    <SelectItem key={value.toString()} value={value.toString()}>
                      {isBooleanField ? (value ? 'Yes' : 'No') : value.toString()}
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </FilterField>
        )
      })}

      {/* Custom column filters */}
      {customColumns.map(col => {
        const options = customColumnOptions[col.id] || []
        if (options.length === 0) return null
        const currentValue = customColumnFilters[col.id] || ''

        return (
          <FilterField key={`custom_${col.id}`} label={col.name} id={`custom_filter_${col.id}`}>
            <Select
              value={currentValue || "any"}
              onValueChange={(value) => {
                const newFilters = { ...customColumnFilters }
                if (value === 'any') {
                  delete newFilters[col.id]
                } else {
                  newFilters[col.id] = value
                }
                onCustomColumnFilterChange?.(newFilters)
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Any" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="any">Any</SelectItem>
                {options.map((val) => (
                  <SelectItem key={val} value={val}>
                    {val}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FilterField>
        )
      })}
    </div>
  )
}