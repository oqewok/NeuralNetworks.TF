using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
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

		public double[] Mask { get; set; }
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

		public double[] GetPointCoord(Rectangle rect)
		{
			var coord = new double[4];

			coord[0] = ConvertFunc(rect.Left, Width);
			coord[1] = ConvertFunc(rect.Top, Heigth);

			//coord[2] = ConvertFunc(rect.Right, Width);
			//coord[3] = ConvertFunc(rect.Top, Heigth);

			//coord[4] = ConvertFunc(rect.Left, Width);
			//coord[5] = ConvertFunc(rect.Bottom, Heigth);

			coord[2] = ConvertFunc(rect.Right, Width);
			coord[3] = ConvertFunc(rect.Bottom, Heigth);

			Mask = coord;

			return coord;
		}

		public Rectangle GetRectangle(double[] coord)
		{
			for(int i = 0; i < coord.Length; i++)
			{
				coord[i] = DeconvertFunc(coord[i], i % 2 == 0 ? Width : Heigth);
			}

			return new Rectangle((int)coord[0], (int)coord[1], (int)Math.Abs(coord[2] - coord[0]), (int)Math.Abs(coord[3] - coord[1]));
		}

		private double ConvertFunc(int x, int a)
		{
			return (double)x / a;
		}

		private double DeconvertFunc(double x, int a)
		{
			return a * x;
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

		public double[] GetRegionsByMask(string filename, int size)
		{
			var maskArr = new double[size];

			var data = File.ReadAllText(filename);
			var splitted = data.Split(new char[] { ' ', '[', ']', ',', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

			int index = 0;
			foreach(var str in splitted)
			{
				var num = double.Parse(str, CultureInfo.GetCultureInfo("en-us"));
				maskArr[index++] = num;
			}

			return maskArr;
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
						mask[maskIndex++] = ratio >= 0.75 ? 1 : 0;
						break;
				}

			}

			return mask;
		}

		public double[] FillNegativeMask()
		{
			return Mask = new double[225];
		}

		public Mat DrawMask(Mat image)
		{
			var mask = Mask;
			var regionInnerColor = Color.Green;
			var regionIntersectColor = Color.Blue;

			var threshold = 0.99;
			var regionRectList = GetRegionsRectangles().ToArray();

			for(int i = 0; i < regionRectList.Length; i++)
			{
				
				if(mask[i] == 1)
				{
					var rect = new Rect(regionRectList[i].X, regionRectList[i].Y, regionRectList[i].Width, regionRectList[i].Height);
					var color = new CvColor(regionInnerColor.R, regionInnerColor.G, regionInnerColor.B);
					image.Rectangle(rect, color);
				}
				else if(mask[i] >= threshold)
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
