using System.Drawing;

namespace ImageConverter
{
	static class RectangleIntersection
	{
		/// <summary> Пересекает ли один прямоугольник другой. </summary>
		public static bool IsIntersects(Rectangle rect1, Rectangle rect2)
		{
			return (rect1.X < rect2.X + rect2.Width && rect1.X + rect1.Width > rect2.X && rect1.Y + rect1.Height > rect2.Y && rect1.Y < rect2.Y + rect2.Height);
		}

		/// <summary> Находится ли один прямоугольник внутри другого. </summary>
		public static bool IsInner(Rectangle bigRect, Rectangle smallRect)
		{
			return ((smallRect.X >= bigRect.X && smallRect.Y >= bigRect.Y) && ((smallRect.X + smallRect.Width <= bigRect.X + bigRect.Width) && (smallRect.Y + smallRect.Height <= bigRect.Y + bigRect.Height)));
		}

		/// <summary> Проверяет расположение прямоугольников относительно друг друга. </summary>
		public static IntersectionResult CheckRectanglesIntersection(Rectangle rect1, Rectangle rect2)
		{
			if(IsInner(rect1, rect2) || IsInner(rect2, rect1))
			{
				return IntersectionResult.Inner;
			}
			else if(!IsIntersects(rect1, rect2))
			{
				return IntersectionResult.Outer;
			}
			else return IntersectionResult.Intersect;
		}

		public static Rectangle Intersection(Rectangle rect1, Rectangle rect2)
		{
			rect1.Intersect(rect2);
			return rect1;
		}

		public static int Square(Rectangle rect)
		{
			return (rect.Right - rect.Left) * (rect.Bottom - rect.Top);
		}

		public static double IntersectionRatio(Rectangle inspectedRect, Rectangle scanningRect)
		{
			var intersection = Intersection(inspectedRect, scanningRect);

			var intersectionSquare = Square(intersection);
			var inspectedRectSquare = Square(inspectedRect);

			return (double)intersectionSquare / inspectedRectSquare;
		}
	}

	public enum IntersectionResult : int
	{
		Inner,
		Intersect,
		Outer
	}
}
