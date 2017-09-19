using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace ImageConverter
{
	public class XmlMarkupExporter
	{
		public const string RootAttributeName = "Image";
		public const string PlatesAttributeName = "Plates";


		private XDocument _xmlFile;

		public XmlMarkupExporter(string fileName)
		{
			FileName = fileName;
			_xmlFile = XDocument.Load(fileName);
		}

		public string FileName { get; set; }

		public IEnumerable<Rectangle> GetPlateRectangles()
		{
			var plates = _xmlFile.Descendants(PlatesAttributeName);

			var humanChecked = _xmlFile
				.Root
				.Elements("HumanChecked")
				.FirstOrDefault();

			if(humanChecked.Attribute("Value").Value == "False")
			{
				//if(plates.Elements("Plate").Count() == 0)
				//{
				//	return null;
				//}
				return null;
			}

			var plateRectangles = new List<Rectangle>();

			foreach(var plate in plates.Elements("Plate"))
			{
				var points = new List<KeyValuePair<int, int>>();

				foreach(var point in plate.Elements("Region").Elements("Point"))
				{
					var x = int.Parse(point.Attribute("X").Value);
					var y = int.Parse(point.Attribute("Y").Value);

					points.Add(new KeyValuePair<int, int>(x, y));
				}

				var xTopLeft = points
					.ElementAt(0)
					.Key;
				var yTopLeft = points
					.ElementAt(0)
					.Value;
				var width = points
					.ElementAt(2)
					.Key - xTopLeft;
				var height = points
					.ElementAt(2)
					.Value - yTopLeft;

				var rect = new Rectangle(xTopLeft, yTopLeft, width, height);
				if(!(rect.X == 0 && rect.Y == 0 && rect.Width == 0 && rect.Height == 0))
					plateRectangles.Add(rect);
			}

			if (plateRectangles.Count != 1)
			{
				return null;
				//plateRectangles.RemoveAll(e => (e.X == 0 && e.Y == 0 && e.Width == 0 && e.Height == 0));
			}

			return plateRectangles;
		}
	}
}
