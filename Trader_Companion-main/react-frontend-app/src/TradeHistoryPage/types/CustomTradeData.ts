// Types for custom trade data tracking

export interface CustomColumn {
  id: number;
  name: string;
  created_at?: string;
}

export interface ColumnOrder {
  id?: number;
  column_key: string;
  position: number;
  is_custom: boolean;
  width?: number; // pixel width, 0 or undefined means use default
  is_textarea?: boolean; // render as expandable textarea
}

export interface CustomColumnValue {
  id?: number;
  trade_id: number;
  column: number; // CustomColumn ID
  value: string;
}
