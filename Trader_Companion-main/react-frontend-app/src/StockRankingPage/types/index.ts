export interface GlobalCharacteristic {
  id: number;
  name: string;
  default_score: number;
  created_at: string;
}

export interface StockCharacteristic {
  id: number;
  name: string;
  score: number;
  characteristic_id: number;
}

export interface StockPick {
  id: number;
  symbol: string;
  total_score: number;
  personal_opinion_score: number;
  demand_reason: string;
  note: string;
  ranking_box: number;
  case_text: string;
  characteristics: StockCharacteristic[];
  created_at: string;
}

export interface RankingBox {
  id: number;
  title: string;
  stock_picks: StockPick[];
  created_at: string;
  order?: number;  // Make order optional since it's only used during drag operations
}

export interface UserPageState {
  column_count: number;
  ranking_boxes_order: number[];
  updated_at: string;
}