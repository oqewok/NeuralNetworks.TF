using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.CPlusPlus;

namespace ImageConverter
{
	public class ImageMaskCreator
	{
		public const int Width = 640;
		public const int Heigth = 480;

		public ImageMaskCreator(int scaleX, int scaleY)
		{
			ScaleX = scaleX;
			ScaleY = scaleY;
		}

		public double[] Mask { get; private set; }
		public int ScaleX { get; set; }
		public int ScaleY { get; set; }

		public string CreateStringMask(Rectangle[] boundBoxes)
		{
			var sb = new StringBuilder();
			var mask = new int[Width, Heigth];

			foreach(var box in boundBoxes)
			{
				for(int i = box.Left; i < box.Right; i++)
				{
					for(int j = box.Top; j < box.Bottom; j++)
					{
						mask[i, j] = 1; 
					}
				}
			}

			for(int i = 0; i < Width; i++)
			{
				for(int j = 0; j < Heigth; j++)
				{
					sb.Append(mask[i, j] + ",");
				}
			}

			return sb.ToString().Substring(0, sb.Length - 1);
		}

		public IEnumerable<Rectangle> GetRegionsRectangles()
		{
			var regionRectList = new List<Rectangle>(ScaleX * ScaleY);
			var regionWidth = (Width / ScaleX) * 2;
			var regionHeight = (Heigth / ScaleY) * 2;

			for(int i = 0; i < Width - regionWidth / 2; i += regionWidth / 2)
			{
				for(int j = 0; j < Heigth - regionWidth / 2; j += regionHeight / 2)
				{
					var regionRect = new Rectangle(i, j, regionWidth, regionHeight);

					regionRectList.Add(regionRect);
				}
			}

			return regionRectList;
		}

		public double[] FillMasks(Rectangle objectRect)
		{
			var regionRectList = GetRegionsRectangles();
			Mask = new double[regionRectList.Count()];
			var mask = Mask;

			int maskIndex = 0;
			foreach(var region in regionRectList)
			{
				var intersectionResult = RectangleIntersection.CheckRectanglesIntersection(objectRect, region);

				switch(intersectionResult)
				{
					case IntersectionResult.Outer:
						maskIndex++;
						break;
					case IntersectionResult.Inner:
						mask[maskIndex++] = 1;
						break;
					case IntersectionResult.Intersect:
						var ratio = RectangleIntersection.IntersectionRatio(region, objectRect);
						mask[maskIndex++] = ratio >= 0.5 ? ratio : 0;
						break;
				}

			}

			return mask;
		}

		public Mat DrawMask(Mat image)
		{
			var mask = Mask;
			var regionInnerColor = Color.Green;
			var regionIntersectColor = Color.Blue;

			var regionRectList = GetRegionsRectangles().ToArray();

			for(int i = 0; i < regionRectList.Length; i++)
			{
				
				if(mask[i] == 1)
				{
					var rect = new Rect(regionRectList[i].X, regionRectList[i].Y, regionRectList[i].Width, regionRectList[i].Height);
					var color = new CvColor(regionInnerColor.R, regionInnerColor.G, regionInnerColor.B);
					image.Rectangle(rect, color);
				}
				else if(mask[i] != 0)
				{
					var rect = new Rect(regionRectList[i].X, regionRectList[i].Y, regionRectList[i].Width, regionRectList[i].Height);
					var color = new CvColor(regionIntersectColor.R, regionIntersectColor.G, regionIntersectColor.B);
					image.Rectangle(rect, color);
				}
				//else
				//{
				//	var rect = new Rect(regionRectList[i].X, regionRectList[i].Y, regionRectList[i].Width, regionRectList[i].Height);
				//	var color = CvColor.Gray;
				//	image.Rectangle(rect, color);
				//}
			}

			return image;
		}

		public override string ToString()
		{
			var sb = new StringBuilder();

			foreach(var elem in Mask)
			{
				sb.Append(elem.ToString(CultureInfo.GetCultureInfo("en-US")) + ",");
			}

			return sb
				.ToString()
				.Substring(0, sb.Length - 1);
		}
	}
}
