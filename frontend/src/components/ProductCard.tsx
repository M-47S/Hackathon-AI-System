import type { ProductCard as ProductType } from '../types/api';
import './ProductCard.css';

interface ProductCardProps {
    product: ProductType;
    onInterested: () => void;
}

export function ProductCard({ product, onInterested }: ProductCardProps) {
    return (
        <div className="product-card">
            <h3>{product.name}</h3>
            <p>{product.description}</p>
            <button className="product-btn" onClick={onInterested}>
                Мне интересно
            </button>
        </div>
    );
}