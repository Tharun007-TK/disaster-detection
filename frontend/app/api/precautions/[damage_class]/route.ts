const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ damage_class: string }> }
) {
  const { damage_class } = await params;
  const res = await fetch(`${API_URL}/api/precautions/${damage_class}`, {
    cache: "no-store",
  });
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
