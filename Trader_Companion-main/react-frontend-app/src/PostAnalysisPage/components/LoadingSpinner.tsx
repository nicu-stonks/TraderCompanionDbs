import { Loader } from "lucide-react";
import React from "react";

const LoadingSpinner: React.FC<{ message?: string }> = ({ message = 'Loading...' }) => (
  <div className="flex items-center justify-center py-8">
    <Loader className="w-6 h-6 animate-spin mr-2 text-gray-600 dark:text-gray-300" />
    <span className="text-gray-600 dark:text-gray-300">{message}</span>
  </div>
);

export default LoadingSpinner;
